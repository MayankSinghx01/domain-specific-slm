import os
import math
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
import time

# =============================================================================
# 1. CREDENTIALS & SENSITIVE KEYS
# =============================================================================
os.environ["WANDB_API_KEY"] = "WANDB_API_KEY"
os.environ["HF_TOKEN"] = "HUGGINGFACE_TOKEN"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0"

# =============================================================================
# 2. HYPERPARAMETERS & CONFIG (PHASE 3)
# =============================================================================
out_dir = "checkpoints/phase3_runs"
os.makedirs(out_dir, exist_ok=True)

# Training targets
target_total_tokens = 4_000_000_000  # Example: 4B tokens for Phase 3
log_interval_steps = 10
eval_every_steps = 200               
save_every_steps = 1000              
max_checkpoints = 3                  
eval_batches = 10                    # Reduced from 20 to lower overhead

# Infra / Batch settings
micro_batch_size = 4                 
sequence_length = 2048               # INCREASED: 1024 -> 2048
global_batch_size = 524288           

# Learning Rate for Phase 3
learning_rate = 2e-5                 # DECREASED: 3e-5 -> 2e-5
min_lr = 2e-6
warmup_iters = 200                   # Short warmup for CPT continuation

# Phase 3 Data Mixture
DATA_MIX_V3 = {
    "openwebmath":     0.38,
    "fineweb_replay":  0.20,
    "numina_math_cot": 0.15,
    "cosmo":           0.10,
    "prime_intellect": 0.10,
    "python_code":     0.05,
    "openthoughts":    0.02,
}

# =============================================================================
# 3. DDP SETUP & AUTO-ITERATION CALCULATION
# =============================================================================
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    master_process = True

# Grad accumulation math (Automatically adapts to 3x Ada RTX 6000s)
tokens_per_iter = micro_batch_size * sequence_length * ddp_world_size
grad_accum_steps = global_batch_size // tokens_per_iter
actual_global_batch = grad_accum_steps * tokens_per_iter

# Auto-calculate Max Iters based on target tokens
max_iters = target_total_tokens // actual_global_batch

if master_process:
    print(f"World Size: {ddp_world_size} GPUs")
    print(f"Sequence Length: {sequence_length}")
    print(f"Tokens per micro-step: {tokens_per_iter:,}")
    print(f"Gradient Accumulation steps: {grad_accum_steps}")
    print(f"Actual tokens per weight update: {actual_global_batch:,}")
    print(f"Calculated Max Iterations: {max_iters}")

torch.set_float32_matmul_precision('high')
ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# =============================================================================
# 4. DATA LOADER (Identical to Phase 2)
# =============================================================================
class ShardManager:
    def __init__(self, prefix, split, seq_len, rank, world_size):
        self.files = glob.glob(os.path.join(f"data/{split}", f"{prefix}_{split}_*.bin"))
        if not self.files:
            raise ValueError(f"No shards found for {prefix} in {split}")
            
        np.random.seed(42 + rank)
        np.random.shuffle(self.files)
        
        self.file_idx = 0
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.data = self._load_shard(self.files[self.file_idx])
        self.ptr = self.rank * self.seq_len 

    def _load_shard(self, path):
        return np.memmap(path, dtype=np.uint16, mode='r', offset=1024)

    def get_sequence(self):
        if self.ptr + self.seq_len + 1 > len(self.data):
            self.file_idx += 1
            if self.file_idx >= len(self.files):
                np.random.shuffle(self.files)
                self.file_idx = 0
            self.data = self._load_shard(self.files[self.file_idx])
            self.ptr = self.rank * self.seq_len

        x = torch.from_numpy((self.data[self.ptr : self.ptr+self.seq_len]).astype(np.int64))
        y = torch.from_numpy((self.data[self.ptr+1 : self.ptr+1+self.seq_len]).astype(np.int64))
        
        self.ptr += self.seq_len * self.world_size 
        return x, y

class EliteDataLoader:
    def __init__(self, split, data_mix, batch_size, seq_len, rank, world_size):
        self.batch_size = batch_size
        self.datasets = list(data_mix.keys())
        weights = list(data_mix.values())
        self.weights = np.array(weights) / np.sum(weights)
        
        np.random.seed(1337 + rank)
        self.managers = {ds: ShardManager(ds, split, seq_len, rank, world_size) for ds in self.datasets}

    def get_batch(self):
        choices = np.random.choice(self.datasets, size=self.batch_size, p=self.weights)
        X, Y = [], []
        for ds in choices:
            x, y = self.managers[ds].get_sequence()
            X.append(x)
            Y.append(y)
        return torch.stack(X).pin_memory().to(device, non_blocking=True), \
               torch.stack(Y).pin_memory().to(device, non_blocking=True)

train_loader = EliteDataLoader('train', DATA_MIX_V3, micro_batch_size, sequence_length, ddp_rank, ddp_world_size)
val_loader = EliteDataLoader('val', DATA_MIX_V3, micro_batch_size, sequence_length, ddp_rank, ddp_world_size)

# =============================================================================
# 5. MODEL ARCHITECTURE & PHASE 2 WEIGHTS INIT
# =============================================================================
from model import IvLLM
model = IvLLM()

# Load Phase 2 Checkpoint
ckpt_path = "checkpoints/phase2_runs/ivllm_phase2_final.pt"
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model = torch.compile(model) 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))

# If resuming mid-phase 3, uncomment to restore optimizer state:
# if "optimizer_state_dict" in checkpoint:
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if master_process:
    wandb.init(
        project="ivllm-phase3",
        name="reasoning-mix-v2",
        config={
            "learning_rate": learning_rate,
            "min_lr": min_lr,
            "global_batch_size": actual_global_batch,
            "sequence_length": sequence_length,
            "max_iters": max_iters,
            "data_mix": DATA_MIX_V3,
        },
    )

# =============================================================================
# 6. EVALUATION FUNCTION (Per-Dataset Granularity)
# =============================================================================
@torch.no_grad()
def estimate_loss():
    """Estimates overall train loss and per-dataset validation loss."""
    out = {}
    model.eval()
    
    # Global Train Loss
    train_losses = torch.zeros(eval_batches) 
    for k in range(eval_batches):
        X, Y = train_loader.get_batch()
        with torch.autocast(device_type='cuda', dtype=ptdtype):
            _, loss = model(X, Y)
        train_losses[k] = loss.item()
    out['train/eval_loss'] = train_losses.mean().item()

    # Per-Dataset Val Loss
    for ds in DATA_MIX_V3.keys():
        val_losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X_batch, Y_batch = [], []
            for _ in range(micro_batch_size):
                x, y = val_loader.managers[ds].get_sequence()
                X_batch.append(x)
                Y_batch.append(y)
            
            X = torch.stack(X_batch).pin_memory().to(device, non_blocking=True)
            Y = torch.stack(Y_batch).pin_memory().to(device, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=ptdtype):
                _, loss = model(X, Y)
            val_losses[k] = loss.item()
            
        out[f"val/{ds}_loss"] = val_losses.mean().item()
        
    model.train()
    return out

# =============================================================================
# 7. MAIN TRAINING LOOP
# =============================================================================
X, Y = train_loader.get_batch()
t0 = time.time()
saved_ckpts = []

for iter_num in range(max_iters):
    
    # LR Schedule: Linear Warmup followed by Cosine Decay
    if iter_num < warmup_iters:
        lr = learning_rate * (iter_num + 1) / warmup_iters
    else:
        decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
        decay_ratio = min(1.0, decay_ratio) 
        lr = min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * decay_ratio))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # -------------------------------------------------------------------------
    # Evaluation Step
    # -------------------------------------------------------------------------
    if iter_num > 0 and iter_num % eval_every_steps == 0 and master_process:
        losses = estimate_loss()
        print(f"\n--- EVAL AT STEP {iter_num} ---")
        for k, v in losses.items():
            print(f"{k}: {v:.4f}")
        print("-----------------------\n")
        wandb.log(losses, step=iter_num)

    # -------------------------------------------------------------------------
    # Checkpointing Step (Compatible with external evaluate.py)
    # -------------------------------------------------------------------------
    if iter_num > 0 and iter_num % save_every_steps == 0 and master_process:
        print(f"\n--- SAVING CHECKPOINT AT STEP {iter_num} ---")
        ckpt_path = os.path.join(out_dir, f'ckpt_{iter_num}.pt')
        
        # Save complete state for resumption, plus weights for external eval scripts
        checkpoint_dict = {
            "model_state_dict": model.module.state_dict() if ddp else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": iter_num
        }
        torch.save(checkpoint_dict, ckpt_path)
        
        saved_ckpts.append(ckpt_path)
        if len(saved_ckpts) > max_checkpoints:
            oldest_ckpt = saved_ckpts.pop(0)
            if os.path.exists(oldest_ckpt):
                os.remove(oldest_ckpt)

    # -------------------------------------------------------------------------
    # Forward / Backward Pass Accumulation Block
    # -------------------------------------------------------------------------
    for micro_step in range(grad_accum_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
        with torch.autocast(device_type='cuda', dtype=ptdtype):
            _, loss = model(X, Y)
            loss = loss / grad_accum_steps 
            
        X, Y = train_loader.get_batch()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # -------------------------------------------------------------------------
    # Logging Step
    # -------------------------------------------------------------------------
    if iter_num % log_interval_steps == 0 and master_process:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        wandb.log({
            "train/loss": loss.item() * grad_accum_steps,
            "train/lr": lr,
            "train/grad_norm": norm,
            "system/dt": dt,
        }, step=iter_num)
        
        print(f"iter {iter_num}: loss {loss.item() * grad_accum_steps:.4f}, step time {dt*1000:.2f}ms")

# =============================================================================
# 8. FINAL INFERENCE-ONLY CHECKPOINT
# =============================================================================
if master_process:
    # Save a clean, weights-only checkpoint for production / deployment inference
    final_checkpoint = {
        "model_state_dict": model.module.state_dict() if ddp else model.state_dict()
    }
    torch.save(final_checkpoint, os.path.join(out_dir, "ivllm_phase3_final_inference.pt"))
    print("Training complete. Final inference checkpoint saved.")

if ddp:
    destroy_process_group()