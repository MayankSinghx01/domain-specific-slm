import os
import glob
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from einops import rearrange, repeat

# =============================================================================
# 1. DDP INITIALIZATION & HARDWARE MAPPING
# =============================================================================
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if master_process:
        print("WARNING: Running on a single GPU. Use torchrun for multi-GPU.")

torch.set_float32_matmul_precision('high')

# =============================================================================
# 2. HYPERPARAMETERS & STREAMING CONFIGURATION
# =============================================================================
# Model Architecture (671M Parameters)
VOCAB_SIZE = 50304         
DIM = 1536                 
NUM_BLOCKS = 24            
NUM_Q_HEADS = 24           
NUM_KV_GROUPS = 6          
QUERIES_PER_GROUP = NUM_Q_HEADS // NUM_KV_GROUPS
HEAD_DIM = DIM // NUM_Q_HEADS 

# Batch Sizing
SEQ_LENGTH = 2048          
MICRO_BATCH_SIZE = 32      
GLOBAL_BATCH_SIZE = 480    
assert GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * ddp_world_size) == 0
GRAD_ACCUM_STEPS = GLOBAL_BATCH_SIZE // (MICRO_BATCH_SIZE * ddp_world_size)

# Streaming Workflow Milestones
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQ_LENGTH
TOTAL_TARGET_TOKENS = 30_000_000_000  
CHUNK_TOKENS = 1_000_000_000          
VAL_TOKENS_PER_CHUNK = 100_000_000    # The first shard we save for validation

GLOBAL_MAX_STEPS = TOTAL_TARGET_TOKENS // TOKENS_PER_STEP
# Only train on the 900M tokens explicitly allocated to the train set
STEPS_PER_CHUNK = (CHUNK_TOKENS - VAL_TOKENS_PER_CHUNK) // TOKENS_PER_STEP

# Learning Rate
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.1

DATA_DIR = "data" 
CHECKPOINT_DIR = "checkpoints"
if master_process:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
# 3. CORE ARCHITECTURE MODULES
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep computation entirely in float32, only downcast at the very end
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq) 
        freqs = torch.cat((freqs, freqs), dim=-1) 
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1)
        
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
            
        q_out = (q * cos) + (rotate_half(q) * sin)
        k_out = (k * cos) + (rotate_half(k) * sin)
        return q_out, k_out

class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(DIM, NUM_Q_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, NUM_KV_GROUPS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, NUM_KV_GROUPS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(NUM_Q_HEADS * HEAD_DIM, DIM, bias=False)
        self.wo.NANOGPT_SCALE_INIT = 1 

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        q = rearrange(self.wq(x), 'b t (h d) -> b h t d', h=NUM_Q_HEADS)
        k = rearrange(self.wk(x), 'b t (g d) -> b g t d', g=NUM_KV_GROUPS)
        v = rearrange(self.wv(x), 'b t (g d) -> b g t d', g=NUM_KV_GROUPS)

        q, k = rope(q, k)

        k = repeat(k, 'b g t d -> b (g r) t d', r=QUERIES_PER_GROUP)
        v = repeat(v, 'b g t d -> b (g r) t d', r=QUERIES_PER_GROUP)

        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        context = rearrange(context, 'b h t d -> b t (h d)')
        return self.wo(context)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = int(2 * (4 * DIM / 3))
        hidden_dim = ((hidden_dim + 127) // 128) * 128 
        self.w_gate_val = nn.Linear(DIM, 2 * hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, DIM, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined = self.w_gate_val(x)
        gate, val = rearrange(combined, 'b t (split h) -> split b t h', split=2)
        return self.w_down(F.silu(gate) * val)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNorm(DIM)
        self.attn = GroupedQueryAttention()
        self.ffn_norm = RMSNorm(DIM)
        self.ffn = SwiGLU()

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class IvLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, DIM)
        self.rope = RoPE(dim=HEAD_DIM, max_seq_len=SEQ_LENGTH)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_BLOCKS)])
        self.final_norm = RMSNorm(DIM)
        self.output_layer = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        
        # Weight Tying
        self.embeddings.weight = self.output_layer.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * NUM_BLOCKS) ** -0.5
            
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        x = self.embeddings(tokens)
        for block in self.blocks:
            x = block(x, self.rope)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(rearrange(logits, 'b t v -> (b t) v'), rearrange(targets, 'b t -> (b t)'))
        return logits, loss

# =============================================================================
# 4. DISTRIBUTED DATALOADER (STREAMING AWARE)
# =============================================================================

class DistributedShardLoader:
    def __init__(self, data_dir: str, split: str, batch_size: int, seq_len: int, rank: int, world_size: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        
        self.files = sorted(glob.glob(os.path.join(data_dir, f"*_{split}_*.bin")))
        if not self.files:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}! Make sure the streaming script ran.")
        
        self.current_shard_idx = 0
        self.position = self.rank * self.batch_size * self.seq_len
        self.load_shard()

    def load_shard(self):
        file_path = self.files[self.current_shard_idx]
        self.mmap = np.memmap(file_path, dtype=np.uint16, mode='r', offset=1024)

    def state_dict(self):
        return {
            'current_shard_idx': self.current_shard_idx,
            'position': self.position
        }

    def load_state_dict(self, state):
        self.current_shard_idx = state['current_shard_idx']
        self.position = state['position'] + (self.rank * self.batch_size * self.seq_len)
        self.load_shard()

    def get_batch(self):
        B, T = self.batch_size, self.seq_len
        tokens_needed = (B * T) + 1
        
        if self.position + tokens_needed > len(self.mmap):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
            self.position = self.rank * B * T
            self.load_shard()
            
        chunk = self.mmap[self.position : self.position + tokens_needed]
        x = torch.from_numpy(chunk[:-1].astype(np.int64)).view(B, T)
        y = torch.from_numpy(chunk[1:].astype(np.int64)).view(B, T)
        
        self.position += (B * T) * self.world_size
        return x.to(device), y.to(device)

# =============================================================================
# 5. TRAINING UTILS
# =============================================================================

def get_lr(global_step: int) -> float:
    # LR Scheduling is based on the GLOBAL 30B horizon
    if global_step < WARMUP_STEPS:
        return MAX_LR * (global_step / WARMUP_STEPS)
    if global_step > GLOBAL_MAX_STEPS:
        return MIN_LR
    decay_ratio = (global_step - WARMUP_STEPS) / (GLOBAL_MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

@torch.no_grad()
def estimate_loss(model, val_loader, eval_steps=20):
    model.eval()

    val_loader.current_shard_idx = 0
    val_loader.position = val_loader.rank * val_loader.batch_size * val_loader.seq_len
    val_loader.load_shard()

    losses = torch.zeros(eval_steps, device=device)
    for k in range(eval_steps):
        x, y = val_loader.get_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[k] = loss
    
    mean_loss = losses.mean()
    if ddp:
        torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.AVG)
        
    model.train()
    return mean_loss.item()

# =============================================================================
# 6. THE STREAMING MAIN LOOP
# =============================================================================

def main():
    if master_process:
        print(f"\n--- IvLLM Streaming Pipeline ---")
        print(f"GPUs: {ddp_world_size}")
        print(f"Global Batch Size: {GLOBAL_BATCH_SIZE} sequences")
        print(f"Target: {STEPS_PER_CHUNK} steps to consume current 1B Chunk")
        print(f"Global Horizon: {GLOBAL_MAX_STEPS} steps to reach 30B Tokens\n")
    
    train_loader = DistributedShardLoader(DATA_DIR, "train", MICRO_BATCH_SIZE, SEQ_LENGTH, ddp_rank, ddp_world_size)
    val_loader = DistributedShardLoader(DATA_DIR, "val", MICRO_BATCH_SIZE, SEQ_LENGTH, ddp_rank, ddp_world_size)
    
    raw_model = IvLLM().to(device)
    if ddp:
        model = DDP(raw_model, device_ids=[ddp_local_rank])
    else:
        model = raw_model
    model = torch.compile(model)
    
    param_dict = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'embeddings' not in n]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or 'embeddings' in n]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=MAX_LR, betas=(0.9, 0.95), fused=torch.cuda.is_available())

    # --- STATE RESUME LOGIC ---
    global_step = 0
    local_step = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "ivllm_latest.pt")
    
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=True)        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        
        if checkpoint.get('chunk_completed', False):
            if master_process:
                print(f"Resuming at Global Step {global_step}. Starting a completely NEW 1B Chunk.")
            # Do NOT load dataloader state. Let it naturally start at index 0 of the new data files.
        else:
            if master_process:
                print(f"Crash detected. Resuming MID-CHUNK at Global Step {global_step}.")
            train_loader.load_state_dict(checkpoint['dataloader_state'])
            local_step = checkpoint['local_step']
    else:
        if master_process:
            print("No checkpoint found. Starting Phase 1 from scratch.")

    model.train()
    t0 = time.time()
    total_tokens_window = 0
    
    # --- TRAINING LOOP (Exits precisely when 1B chunk is done) ---
    while local_step < STEPS_PER_CHUNK and global_step < GLOBAL_MAX_STEPS:
        lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.get_batch()
            sync_context = model.no_sync() if ddp and micro_step < GRAD_ACCUM_STEPS - 1 else nullcontext()
            
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM_STEPS
                loss_accum += loss.detach().item()
                loss.backward()
            
            total_tokens_window += (MICRO_BATCH_SIZE * SEQ_LENGTH * ddp_world_size)

        if ddp:
            loss_tensor = torch.tensor(loss_accum, device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            loss_accum = loss_tensor.item()

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Step tracking
        global_step += 1
        local_step += 1

        if master_process and global_step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = total_tokens_window / dt
            print(f"Global Step: {global_step:6d} | Local: {local_step}/{STEPS_PER_CHUNK} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")
            t0 = time.time()
            total_tokens_window = 0

        # Mid-chunk safety backups
        if local_step < STEPS_PER_CHUNK and local_step % 500 == 0:
            val_loss = estimate_loss(model, val_loader)
            if master_process:
                print(f"--- Mid-Chunk Backup (Val Loss: {val_loss:.4f}) ---")
                
                # --- NEW LOGIC: Rotate the old latest to prev ---
                prev_ckpt = os.path.join(CHECKPOINT_DIR, "ivllm_prev.pt")
                if os.path.exists(latest_ckpt):
                    os.replace(latest_ckpt, prev_ckpt)
                # ------------------------------------------------
                
                tmp_ckpt = latest_ckpt + ".tmp"
                torch.save({
                    'global_step': global_step,
                    'local_step': local_step,
                    'chunk_completed': False,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataloader_state': train_loader.state_dict(),
                }, tmp_ckpt)
                os.replace(tmp_ckpt, latest_ckpt)

    # --- CHUNK COMPLETION EXIT ---
    if master_process:
        print("\n===========================================")
        print(f"SUCCESS: 1 Billion Token Chunk Consumed!")
        print(f"Securing final state at Global Step {global_step}...")
        
        # Save state with chunk_completed=True so the next run knows to look for fresh data
        tmp_ckpt = latest_ckpt + ".tmp"
        torch.save({
            'global_step': global_step,
            'local_step': 0, 
            'chunk_completed': True, 
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, tmp_ckpt)
        
        os.replace(tmp_ckpt, latest_ckpt)

        print("Data safely backed up. You may now delete the .bin files in /data and stream the next batch.")
        print("===========================================\n")

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()