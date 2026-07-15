import os
import json
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# =============================================================================
# 1. STREAMING CONFIGURATION
# =============================================================================
TARGET_TOKENS = 1_000_000_000  # 1 Billion tokens per extraction run
SHARD_SIZE = 100_000_000       # 100M tokens per file
DATA_DIR = "data"              # Must match train.py exactly
STATE_FILE = "fineweb_state.json"
REMOTE_NAME = "sample-100BT"   

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize GPT-2 Tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

# =============================================================================
# 2. FILE WRITING UTILS
# =============================================================================
def write_datafile(filename, toks):
    """Saves token data as a .bin file, for reading natively in PyTorch."""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens
    
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
        
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

# =============================================================================
# 3. MAIN EXTRACTION ENGINE
# =============================================================================
def main():
    skip_docs = 0
    run_index = 0
    
    # Load bookmark state if it exists
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            skip_docs = state.get("skip_docs", 0)
            run_index = state.get("run_index", 0)

    print(f"\n--- Starting Data Extraction Phase (Run #{run_index + 1}) ---")
    if skip_docs > 0:
        print(f"Fast-forwarding past {skip_docs:,} previously trained documents...")

    # STREAMING=TRUE prevents local disk caching of the massive dataset
    fw = load_dataset("HuggingFaceFW/fineweb", name=REMOTE_NAME, split="train", streaming=True)
    fw = fw.skip(skip_docs)

    nprocs = max(1, os.cpu_count() - 2)
    shard_index = 0 # Always reset to 0 so the dataloader reads cleanly
    
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    total_tokens_extracted = 0
    docs_processed = 0
    
    progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Writing Shard {shard_index:02d}")

    # Process documents as they arrive over the network
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            docs_processed += 1
            
            # If this document pushes us over the 1B limit, truncate it to fit exactly
            if total_tokens_extracted + len(tokens) >= TARGET_TOKENS:
                tokens = tokens[:TARGET_TOKENS - total_tokens_extracted]

            if token_count + len(tokens) < SHARD_SIZE:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                total_tokens_extracted += len(tokens)
                progress_bar.update(len(tokens))
            else:
                # The shard buffer is full, write it to disk
                remainder = SHARD_SIZE - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                
                # Shard 0 is set aside as validation split
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
                write_datafile(filename, all_tokens_np)
                
                shard_index += 1
                total_tokens_extracted += remainder
                
                if total_tokens_extracted >= TARGET_TOKENS:
                    progress_bar.close()
                    break
                
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Writing Shard {shard_index:02d}")
                
                # Carry the leftover tokens over to the new empty shard
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
                total_tokens_extracted += token_count
                progress_bar.update(token_count)

            if total_tokens_extracted >= TARGET_TOKENS:
                progress_bar.close()
                break

    # [NEW BLOCK] Flush the final partial shard to disk so we don't lose the last <100M tokens
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
        print(f"Flushed final partial shard: {token_count:,} tokens.")

    # Save the bookmark for the next iteration
    new_state = {
        "skip_docs": skip_docs + docs_processed,
        "run_index": run_index + 1
    }
    with open(STATE_FILE, "w") as f:
        json.dump(new_state, f)
        
    print(f"\nExtraction Complete! Saved {total_tokens_extracted:,} tokens to ./{DATA_DIR}/")
    print(f"State saved. Ready for training.\n")

if __name__ == "__main__":
    main()