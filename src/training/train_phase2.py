import os
import math
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import GPT2TokenizerFast
import wandb
import time

# =============================================================================
# [USER ACTION REQUIRED] 1. CREDENTIALS & SENSITIVE KEYS
# =============================================================================
# Injected directly into the environment so it won't overwrite global server files
os.environ["WANDB_API_KEY"] = "WANDB_API_KEY"
os.environ["HF_TOKEN"] = "HUGGINGFACE_TOKEN"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0"

# =============================================================================
# 2. HYPERPARAMETERS & CONFIG
# =============================================================================
out_dir = "checkpoints/phase2_runs"
os.makedirs(out_dir, exist_ok=True)

# Training targets
eval_interval_tokens = 1_000_000_000 # Benchmark every 1B tokens processed globally
log_interval_steps = 10              # Log to WandB every 10 steps
max_checkpoints = 3                  # Keep only the latest 3 checkpoints

# Infra / Batch settings
micro_batch_size = 4                 # Per-GPU batch size
sequence_length = 2048               # Context length
global_batch_size = 524288           # Target ~0.5M tokens per weight update

# Learning Rate for Phase 2
learning_rate = 3e-5
min_lr = 3e-6
max_iters = 7629                    # Adjust based on total token budget
warmup_iters = 500                   # Linear warmup steps before cosine decay

# Optimized Data Mixing Weights
DATA_MIX = {
    "reasoning_train":       0.35,
    "prime_intellect_train": 0.15,
    "latex_train":           0.15,
    "cosmo_train":           0.15,
    "python_code_train":     0.10,
    "fineweb_replay_train":  0.10,
}

# =============================================================================
# 3. DDP SETUP (Dynamic GPU Handling)
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

# Gradient Accumulation Calculation
tokens_per_iter = micro_batch_size * sequence_length * ddp_world_size
grad_accum_steps = global_batch_size // tokens_per_iter

if master_process:
    print(f"World Size: {ddp_world_size} GPUs")
    print(f"Tokens per micro-step: {tokens_per_iter:,}")
    print(f"Gradient Accumulation steps: {grad_accum_steps}")
    print(f"Total tokens per weight update: {tokens_per_iter * grad_accum_steps:,}")

torch.set_float32_matmul_precision('high')
ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# =============================================================================
# 4. ELITE DATA LOADER (DDP-Patched: Zero Overlap, Perfect Shard Cycling)
# =============================================================================
class ShardManager:
    """Manages reading a single dataset linearly, correctly sharded across DDP ranks."""
    def __init__(self, prefix, split, seq_len, rank, world_size):
        
        if split == "val":
            file_prefix = prefix.replace("_train", "_val")
        else:
            file_prefix = prefix

        self.files = glob.glob(
            os.path.join(f"data/{split}", f"{file_prefix}_*.bin")
        )

        if not self.files:
            raise ValueError(f"No shards found for {file_prefix} in {split}")
            
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

train_loader = EliteDataLoader('train', DATA_MIX, micro_batch_size, sequence_length, ddp_rank, ddp_world_size)
val_loader = EliteDataLoader('val', DATA_MIX, micro_batch_size, sequence_length, ddp_rank, ddp_world_size)

# =============================================================================
# 5. MODEL ARCHITECTURE & PHASE 1 WEIGHTS INIT
# =============================================================================
from model import IvLLM

model = IvLLM()

ckpt_path = "checkpoints/phase1_base/ivllm_latest.pt"
checkpoint = torch.load(
    ckpt_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model = torch.compile(model) 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Initialize Legacy GPT2 Tokenizer for live validation benchmarks
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

if master_process:
    wandb.init(
        project="ivllm-phase2",
        name="reasoning-mix-v1",
        config={
            "learning_rate": learning_rate,
            "min_lr": min_lr,
            "global_batch_size": global_batch_size,
            "micro_batch_size": micro_batch_size,
            "sequence_length": sequence_length,
            "grad_accum_steps": grad_accum_steps,
            "data_mix": DATA_MIX,
        },
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))

# =============================================================================
# 6. EVALUATION & STANDALONE GENERATION FUNCTIONS
# =============================================================================
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(20) 
        for k in range(20):
            X, Y = loader.get_batch()
            with torch.autocast(device_type='cuda', dtype=ptdtype):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate_text(model_instance, idx, max_new_tokens, block_size=2048):
    """Standalone autoregressive generation loop."""
    model_instance.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model_instance(idx_cond)
        logits = logits[:, -1, :] 
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    model_instance.train()
    return idx

@torch.no_grad()
def generate_benchmark():
    prompt = "Question: What is the derivative of x^2 + 2x?\nThought process:"
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    raw_model = model.module if ddp else model
    generated_tokens = generate_text(raw_model, tokens, max_new_tokens=64, block_size=sequence_length)
    result_text = tokenizer.decode(generated_tokens[0])
    return result_text

# =============================================================================
# 7. MAIN TRAINING LOOP
# =============================================================================
X, Y = train_loader.get_batch()
t0 = time.time()
tokens_processed_since_eval = 0
saved_ckpts = []

for iter_num in range(max_iters):
    
    # -------------------------------------------------------------------------
    # LR Schedule: Linear Warmup followed by Cosine Decay
    # -------------------------------------------------------------------------
    if iter_num < warmup_iters:
        lr = learning_rate * (iter_num + 1) / warmup_iters
    else:
        decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
        decay_ratio = min(1.0, decay_ratio) 
        lr = min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * decay_ratio))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # -------------------------------------------------------------------------
    # Eval, Live Generation and Rolling Checkpoint Management
    # -------------------------------------------------------------------------
    if iter_num > 0 and tokens_processed_since_eval >= eval_interval_tokens and master_process:
        losses = estimate_loss()
        gen_text = generate_benchmark()
        
        print(f"\n--- EVAL AT STEP {iter_num} ---")
        print(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"Benchmark Sample Output:\n{gen_text}\n-----------------------\n")
        
        wandb.log({
            "val/loss": losses['val'],
            "train/eval_loss": losses['train'],
            "benchmark_sample": wandb.Html(f"<pre>{gen_text}</pre>")
        }, step=iter_num)
        
        ckpt_path = os.path.join(out_dir, f'ckpt_{iter_num}.pt')
        checkpoint = {
    "model_state_dict": model.module.state_dict() if ddp else model.state_dict()
}
        torch.save(checkpoint, ckpt_path)
        
        saved_ckpts.append(ckpt_path)
        if len(saved_ckpts) > max_checkpoints:
            oldest_ckpt = saved_ckpts.pop(0)
            if os.path.exists(oldest_ckpt):
                os.remove(oldest_ckpt)
                
        tokens_processed_since_eval = 0

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

    tokens_processed_since_eval += (tokens_per_iter * grad_accum_steps)

    # -------------------------------------------------------------------------
    # Periodic Performance Metrics Output
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

if master_process:
    torch.save(
        {
            "model_state_dict": model.module.state_dict() if ddp else model.state_dict()
        },
        os.path.join(out_dir, "ivllm_phase2_final.pt")
    )

if ddp:
    destroy_process_group()