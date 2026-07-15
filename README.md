# IvLLM: A Mathematical Reasoning Language Model

## Overview
IvLLM is a **Mathematical Reasoning Language Model** made from scratch, progressing from foundational sequence models to a ~671M parameter, **OLMo-style** architecture. This project tracks the development of a custom language model from scratch. Rather than just calling APIs, the focus is on building and understanding sequential modeling and modern LLM architecture.

## Model Architecture Specifications
The core architecture is implemented completely from scratch using highly optimized PyTorch primitives:
*   **Parameter Scale:** ~671M parameters (24 layers, hidden dimension of 1536).
*   **Attention Mechanism:** Grouped Query Attention (GQA) with 24 Query heads and 6 Key-Value groups (4 queries per group) to optimize memory utilization during KV-cache generation.
*   **Positional Embedding:** Rotary Position Encodings (RoPE) applied at each attention block.
*   **Activation & Normalization:** Bias-free SwiGLU blocks combined with Root Mean Square Normalization (RMSNorm).
*   **Weight Tying:** Shared token embedding and final output projection matrix layers.

---

## Three-Phase Training Pipeline

### Phase 1: Base Pre-training Engine
*   **Objective:** Develop core linguistic understanding and contextual understanding of the language.
*   **Dataset:** Scaled streaming of 18B FineWeb-Edu tokens (`sample-10BT`) split into 100M token chunk shards (`uint16`).
*   **Precision:** Distributed Data Parallel (DDP) utilizing BF16 mixed-precision, resulting in 50% memory footprint reduction.
*   **Context Window:** 2048 sequence tokens using legacy GPT-2 tokenizer.

### Phase 2: Mathematical Reasoning & Adaptation
*   **Objective:** Align model capabilities towards formal mathematical expressions, proofs, and syntax structures.
*   **Data Mix Strategy:** Linear sampling from an optimized multi-domain corpus:
    *   Reasoning Instruction Corpus: 35%
    *   Verifiable Math Problems (Prime Intellect) & LaTeX Sets: 30%
    *   Cosmopedia & Technical Replays: 25%
    *   Python Source Code: 10%
*   **Training Setup:** Cosine decay scheduling initialized directly from Phase 1 checkpoints.

### Phase 3: Extended Continuation Pre-training (CPT)
*   **Objective:** Solidify multi-step chain-of-thought mathematical solutions while expanding token scale limits.
*   **Data Mix Strategy:** Upgraded data distribution targeted at a 4B token extended horizon.
    *   OpenWebMath: 38%
    *   FineWeb-Edu Replay: 20%
    *   NuminaMath-CoT: 15%
    *   Cosmopedia Auto-Math & Prime Intellect: 20%
    *   Python Code & OpenThoughts (Filtered to < 1800 token spans): 7%
*   **Hyperparameters:** Configured with an expanded context limit of 2048 sequences and a conservative base learning rate ($2 \times 10^{-5}$) to ensure architectural stability during fine-tuning.

## Hardware & Compute
The model was trained on a distributed compute cluster utilizing **3x NVIDIA RTX 6000 Ada Generation GPUs**. To maximize hardware throughput, the training pipeline heavily leverages Distributed Data Parallel (DDP) and BF16 mixed precision, scaling the global batch size efficiently across the multi-GPU setup.

## Current Training Results
*Note: The model is currently a foundational base model. It excels at continuing mathematical texts and proofs but has not yet been instruction-tuned for Q&A.*

**Phase 3 (Continuation Pre-Training) Metrics:**
*   **Final Training Loss:** ~1.99
*   **Final Evaluation Loss:** 1.82
*   **Gradient Norm Stabilization:** Maintained at ~0.38

---

## Tracking & Visualization

All training runs are actively instrumented using Weights & Biases for validation telemetry. 

*   **Project Workspace:** [View IvLLM WandB Project Dashboard](https://wandb.ai/mayanksingh-x01-visvesvaraya-national-institute-of-techn/ivllm-phase3)
*   **Core Tracked Metrics:**
    *   `train/loss` & `val/loss` convergence profiles.
    *   Granular per-dataset cross-entropy validation tracking (`val/openwebmath_loss`, etc.).
    *   Hardware throughput parameters (`system/dt`, tokens per second processing speeds).
    *   Autoregressive validation tracking text streams (`benchmark_sample`).

## Project Repository Map
Refer to the codebase directory configuration below to navigate the implementation steps across the data extraction engines, core model blocks, and execution scripts.

```text
IvLLM/
├── README.md
├── requirements.txt
└── src/
    ├── data/
    │   ├── extract_fineweb.py       # Phase 1 FineWeb token extraction engine
    │   ├── preprocess_math.py       # Phase 2 Mathematical data sharding script
    │   └── preprocess_phase3.py     # Phase 3 Advanced data mix streaming script
    ├── model/
    │   ├── __init__.py
    │   └── ivllm.py                 # Full 671M structural model definition & Phase 1 trainer
    └── training/
        ├── train_phase2.py          # Phase 2 Domain Adaptation execution routine
        └── train_phase3.py          # Phase 3 Extended CPT training loop
