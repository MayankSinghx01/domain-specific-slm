import os
import hashlib
import numpy as np
import traceback
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
from itertools import cycle


# =============================================================================
# 1. CONFIGURATION & PHASE 3 TARGETS
# =============================================================================
TOKENIZER_NAME = "gpt2"
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TOKENS_PER_SHARD = 100_000_000  # 100M tokens per shard

MAGIC_NUMBER = 20240520
VERSION = 1
HEADER_INTS = 256

# Total Phase 3 Budget
TARGET_TOTAL_TOKENS = 4_000_000_000 

# Phase 3 Mixture Percentages
DATA_MIX_V3 = {
    "openwebmath":     0.38,
    "fineweb_replay":  0.20,
    "numina_math_cot": 0.15,
    "cosmo":           0.10,
    "prime_intellect": 0.10,
    "python_code":     0.05,
    "openthoughts":    0.02,
}

# Configurable Python dataset path (Change this to match your exact Phase 2 corpus)
PYTHON_DATASET_PATH = "bigcode/starcoderdata"
PYTHON_DATASET_SUBSET = "python"

# (Prefix, HF_Dataset_Path, Subset, Split)
DATASETS = {
    "numina_math_cot": ("AI-MO/NuminaMath-CoT", None, "train"),
    "openthoughts": ("open-thoughts/OpenThoughts-114k", None, "train"),
    "openwebmath": ("open-web-math/open-web-math", None, "train"),
    "fineweb_replay": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
    "cosmo": ("HuggingFaceTB/cosmopedia", "auto_math_text", "train"),
    "python_code": (PYTHON_DATASET_PATH, PYTHON_DATASET_SUBSET, "train"), 
    "prime_intellect": ("PrimeIntellect/verifiable-math-problems", None, "train"),
}

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
EOT_TOKEN = tokenizer.eos_token_id

stats = defaultdict(lambda: {"train_tok": 0, "val_tok": 0})

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def write_shard(filename, tokens_list):
    """Writes tokens using Phase 2 exact binary header format."""
    tokens_arr = np.array(tokens_list, dtype=np.uint16)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC_NUMBER
    header[1] = VERSION
    header[2] = len(tokens_arr)
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_arr.tobytes())

def extract_text(prefix, row):
    """Handles dataset-specific schemas to reconstruct pure text prior to tokenization."""
    if prefix == "numina_math_cot":
        return f"Problem: {row.get('problem', '')}\nSolution: {row.get('solution', '')}"

    elif prefix == "openthoughts":
        # Preserve the system prompt while keeping Phase 2-style conversation formatting
        parts = []
        if row.get("system"):
            parts.append(row["system"])

        convs = row.get("conversations", [])
        parts.extend([c.get("value", "") for c in convs if c.get("value")])

        return "\n\n".join(parts)
    elif prefix == "prime_intellect":
        return row.get("prompt", "") + "\n\n" + row.get("gold_standard_solution", "")
    elif prefix == "python_code":
        return row.get("content", "")
    else:
        return row.get("text", "")

# =============================================================================
# 3. STREAMING & SHARDING PROCESSOR
# =============================================================================
def process_dataset(prefix, dataset_path, subset_name, split, target_tokens):
    print(f"\n--- Starting {prefix} stream ({target_tokens:,} tokens targeted) ---")
    
    # Load with streaming (trust_remote_code removed as per modern HF standards)
    if subset_name:
        dataset = load_dataset(dataset_path, name=subset_name, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_path, split=split, streaming=True)
        
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    if prefix == "numina_math_cot":
        dataset = cycle(dataset)
    
    train_tokens, val_tokens = [], []
    train_shard_idx, val_shard_idx = 0, 0
    total_train_processed = 0
    
    pbar = tqdm(total=target_tokens, desc=f"Tokenizing {prefix}")
    
    for row in dataset:
        text = extract_text(prefix, row)
        if not text: continue
            
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Specific filtering step for OpenThoughts (< 1800 GPT-2 tokens)
        if prefix == "openthoughts" and len(tokens) > 1800:
            continue
            
        tokens.append(EOT_TOKEN)
        
        # Deterministic Train/Val split based on hash content (approx 99/1 split)
        if int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 100 == 0:
            val_tokens.extend(tokens)
            stats[prefix]["val_tok"] += len(tokens)
        else:
            train_tokens.extend(tokens)
            
        # Write Shards when buffers are full
        while len(train_tokens) >= TOKENS_PER_SHARD:
            remaining = target_tokens - total_train_processed
            if remaining <= 0: break
            
            slice_size = min(TOKENS_PER_SHARD, remaining)
            shard = train_tokens[:slice_size]
            train_tokens = train_tokens[slice_size:]
            write_shard(os.path.join(TRAIN_DIR, f"{prefix}_train_{train_shard_idx:03d}.bin"), shard)
            train_shard_idx += 1
            total_train_processed += slice_size
            stats[prefix]["train_tok"] += slice_size
            pbar.update(slice_size)
            
        while len(val_tokens) >= TOKENS_PER_SHARD:
            shard = val_tokens[:TOKENS_PER_SHARD]
            val_tokens = val_tokens[TOKENS_PER_SHARD:]
            write_shard(os.path.join(VAL_DIR, f"{prefix}_val_{val_shard_idx:03d}.bin"), shard)
            val_shard_idx += 1

        # Check early stopping condition (Quotas)
        if total_train_processed >= target_tokens: 
            break

    # Flush remaining tokens to disk
    remaining = target_tokens - total_train_processed
    if train_tokens and remaining > 0:
        slice_size = min(len(train_tokens), remaining)
        write_shard(os.path.join(TRAIN_DIR, f"{prefix}_train_{train_shard_idx:03d}.bin"), train_tokens[:slice_size])
        total_train_processed += slice_size
        stats[prefix]["train_tok"] += slice_size
        pbar.update(slice_size)
        
    if val_tokens:
        write_shard(os.path.join(VAL_DIR, f"{prefix}_val_{val_shard_idx:03d}.bin"), val_tokens)
        
    pbar.close()

    # Quota verification & warning
    if total_train_processed < target_tokens:
        print(f"\n[WARNING] Dataset '{prefix}' exhausted before reaching target token budget!")
        print(f"  Target Budget (Train): {target_tokens:,}")
        print(f"  Actual Train Tokens:   {stats[prefix]['train_tok']:,}")
        print(f"  Actual Val Tokens:     {stats[prefix]['val_tok']:,}\n")

if __name__ == "__main__":
    for prefix, percentage in DATA_MIX_V3.items():
        dataset_args = DATASETS[prefix]
        target_tokens = int(TARGET_TOTAL_TOKENS * percentage)
        
        try:
            process_dataset(prefix, *dataset_args, target_tokens)
        except Exception:
            print(f"Error processing {prefix}:")
            traceback.print_exc()

    # =============================================================================
    # 4. FINAL VERIFICATION SUMMARY
    # =============================================================================
    print("\n" + "="*80)
    print(f"{'Dataset':<20} | {'Target Tokens':<15} | {'Actual Train Tokens':<20} | {'Actual Val Tokens':<15}")
    print("="*80)
    total_train = 0
    total_val = 0
    for name, p in DATA_MIX_V3.items():
        target = int(TARGET_TOTAL_TOKENS * p)
        t_tok = stats[name]["train_tok"]
        v_tok = stats[name]["val_tok"]
        total_train += t_tok
        total_val += v_tok
        print(f"{name:<20} | {target:<15,} | {t_tok:<20,} | {v_tok:<15,}")
    print("-"*80)
    print(f"{'TOTAL':<20} | {TARGET_TOTAL_TOKENS:<15,} | {total_train:<20,} | {total_val:<15,}")
    print("="*80)
    print("\nData preparation complete!")