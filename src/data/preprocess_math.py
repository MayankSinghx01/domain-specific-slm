import os
import hashlib
import numpy as np
import traceback
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TOKENIZER_NAME = "gpt2"
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TOKENS_PER_SHARD = 100_000_000  # 100M tokens per shard

MAGIC_NUMBER = 20240520
VERSION = 1
HEADER_INTS = 256

# Only list the datasets you still need to process.
# The script will skip ones where shards already exist if you add a check,
# but for now, just keep the ones you want to run.
DATASETS = [
    ("prime_intellect", "PrimeIntellect/verifiable-math-problems", None,
     "train", "question", 300_000_000, 0),
]

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
EOT_TOKEN = tokenizer.eos_token_id

def write_shard(filename, tokens_list):
    tokens_arr = np.array(tokens_list, dtype=np.uint16)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC_NUMBER
    header[1] = VERSION
    header[2] = len(tokens_arr)
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_arr.tobytes())

def process_dataset(prefix, dataset_path, subset_name, split, text_column, target_tokens, skip_rows):

    # Check if we already finished this dataset to save time
    existing_files = [f for f in os.listdir(TRAIN_DIR) if f.startswith(prefix)]
    if existing_files:
        print(f"Skipping {prefix} (already exists).")
        return

    print(f"\n--- Starting {prefix} stream ({target_tokens:,} tokens targeted) ---")
    
    dataset = load_dataset(dataset_path, name=subset_name, split=split, streaming=True) if subset_name else load_dataset(dataset_path, split=split, streaming=True)
        
    if skip_rows > 0:
        dataset = dataset.skip(skip_rows)
    
    train_tokens, val_tokens = [], []
    train_shard_idx, val_shard_idx = 0, 0
    total_train_processed = 0
    
    pbar = tqdm(total=target_tokens, desc=f"Tokenizing {prefix}")
    
    for row in dataset:
        if prefix == "reasoning":
            convs = row.get("conversations", [])
            text = "\n".join([c.get("value", "") for c in convs])
        elif prefix == "prime_intellect":
            text = (
                row.get("prompt", "")
                + "\n\n"
                + row.get("gold_standard_solution", "")
            )
        elif prefix == "python_code":
            text = row.get("content", "")
        else:
            text = row.get(text_column, "")
            
        if not text: continue
            
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(EOT_TOKEN)
        
        if int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 100 == 0:
            val_tokens.extend(tokens)
        else:
            train_tokens.extend(tokens)
            
        while len(train_tokens) >= TOKENS_PER_SHARD:
            remaining = target_tokens - total_train_processed
            if remaining <= 0: break
            slice_size = min(TOKENS_PER_SHARD, remaining)
            shard = train_tokens[:slice_size]
            train_tokens = train_tokens[slice_size:]
            write_shard(os.path.join(TRAIN_DIR, f"{prefix}_train_{train_shard_idx:03d}.bin"), shard)
            train_shard_idx += 1
            total_train_processed += slice_size
            pbar.update(slice_size)
            
        while len(val_tokens) >= TOKENS_PER_SHARD:
            shard = val_tokens[:TOKENS_PER_SHARD]
            val_tokens = val_tokens[TOKENS_PER_SHARD:]
            write_shard(os.path.join(VAL_DIR, f"{prefix}_val_{val_shard_idx:03d}.bin"), shard)
            val_shard_idx += 1

        if total_train_processed >= target_tokens: break

    remaining = target_tokens - total_train_processed
    if train_tokens and remaining > 0:
        slice_size = min(len(train_tokens), remaining)
        write_shard(os.path.join(TRAIN_DIR, f"{prefix}_train_{train_shard_idx:03d}.bin"), train_tokens[:slice_size])
        pbar.update(slice_size)
        
    if val_tokens:
        write_shard(os.path.join(VAL_DIR, f"{prefix}_val_{val_shard_idx:03d}.bin"), val_tokens)
        
    pbar.close()

if __name__ == "__main__":
    for p in DATASETS:
        try:
            process_dataset(*p)
        except Exception:
            traceback.print_exc()
    print("\nDone!")