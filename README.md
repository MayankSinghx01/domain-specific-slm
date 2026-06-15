# Domain-Specific Small Language Model (SLM)

## Aim
To engineer a **Domain-Specific Small Language Model** from scratch, progressing from foundational sequence models to a ~1.5B parameter, **OLMo-style** sparse Mixture-of-Experts (MoE) architecture.

---

## Project Overview
This project tracks the development of a custom language model from scratch. Rather than just calling APIs, the focus is on building and understanding sequential modeling and modern LLM architecture.

The project timeline begins with basic attention-based encoder–decoder implementations and benchmarking (RNN/LSTM) and later moves onto a custom **decoder-only Transformer** heavily inspired by OLMo and LLaMA. 

Key architectural choices include:
- **Sparse MoE:** 8 experts with top-2 routing in the FFNN to increase model capacity without having a very high count of active parameters per token.
- **Attention:** Grouped Query Attention (GQA) to reduce KV-cache memory overload applied along with Rotary Position Encodings (RoPE).
- **Efficiency:** Bias-free SwiGLU activations, RMSNorm, and BF16 precision training.

Following base pre-training on a custom BPE tokenizer, the model will undergo Supervised Fine-Tuning (SFT) using parameter-efficient methods like **QLoRA**. The final stages target preference alignment (DPO/GRPO) and INT4/8 quantization for hardware-constrained deployment on a Raspberry Pi 5.

---

## Current Progress

### Completed
- Implemented Next Word Predictor from scratch using basic RNN and LSTM blocks.
- Implemented attention-based encoder–decoder architecture for sequence tasks.
- Engineered foundational Transformer components (multi-head self-attention, masked causal attention).
- Designed the $\sim$1.5B parameter OLMo-style MoE architecture and custom BPE tokenizer.

### In Progress
- Executing base pre-training pipeline in BF16 precision.
- Setting up infrastructure for Supervised Fine-Tuning (SFT) using QLoRA.

---

## Foundational Work (Brief)
Before tackling language models, the following were implemented to solidify fundamentals in backpropagation, optimization, feature extraction, and sequence-to-sequence architectures:
- ANN from scratch using NumPy (MNIST classification)
- ANN using PyTorch
- CNN using PyTorch (Fashion MNIST)
- RNNs using Pytorch (Next word predictor)
- LSTMs and Attention-Based Encoder-Decoder using PyTorch (Next word predictor)

---

## Project Progress Tracker

### Phase 1: Sequence Models
- [x] RNN implementation  
- [x] LSTM implementation  
- [x] Next-word prediction baseline  

### Phase 2: Attention
- [x] Attention mechanism  
- [x] Encoder–Decoder architecture  

### Phase 3: Base Transformers
- [x] Self-attention (scaled dot-product)  
- [x] Multi-head attention  
- [x] Positional encoding  
- [x] Standard Transformer block  

### Phase 4: OLMo-Style MoE Architecture (IvLLM)
- [x] Decoder-only architecture  
- [x] Custom BPE Tokenizer integration
- [x] Grouped Query Attention (GQA) & RoPE
- [x] Sparse MoE FFNN (Top-2 routing, SwiGLU) 
- [ ] BF16 Pre-training  

### Phase 5: Optimization & Fine-Tuning
- [ ] Supervised Fine-Tuning (SFT)
- [ ] Parameter-Efficient Fine-Tuning (QLoRA)  
- [ ] INT4/8 Quantization

### Phase 6: Preference Alignment
- [ ] Proximal Policy Optimization (PPO) - *Baseline*
- [ ] Direct Preference Optimization (DPO)
- [ ] Group Relative Policy Optimization (GRPO)

### Phase 7: Deployment
- [ ] Edge deployment & inference optimization (Raspberry Pi 5)  

---

## Learning Roadmap

1. RNN → LSTM → Attention  
2. Encoder–Decoder Architectures  
3. Standard Transformer Architecture  
4. **OLMo-style Decoder-only Transformer (MoE, GQA, RoPE)**  
5. Base Pre-training & BPE Tokenization
6. SFT & Parameter-Efficient Fine-Tuning (QLoRA)  
7. Preference Alignment via Deep Reinforcement Learning (DPO/GRPO) 
8. Model Quantization & Edge Deployment

---

## Experiments 

| Model | Status | Notes |
|------|--------|------|
| **RNN** | Completed | Baseline sequence model; severe vanishing gradient issues. |
| **LSTM** | Completed | Handled long-term dependencies significantly better than RNN. |
| **Attention LSTM** | Completed | Improved contextual learning and sequence translation. |
| **Vanilla Transformer** | Completed | Generated dynamic contextual embeddings; achieved GPU parallelization. |
| **Domain-Specific SLM** | In Progress | Final objective. Sparse MoE, RoPE, GQA. Currently in base pre-training phase. |

---

## Tech Stack
- **Languages and Frameworks:** Python, PyTorch, NumPy
- **Tokenization:** Hugging Face Tokenizers (Custom BPE)
- **Training Optimization:** BF16 Mixed Precision, AdamW
- **Supervised Fine-Tuning:** PEFT (QLoRA), Model Quantization (INT4/8)
- **Alignment/RL:** DPO, GRPO, PPO

---

## References

1. Sutskever, I., Vinyals, O., and Le, Q. V.  
   *Sequence to Sequence Learning with Neural Networks* (2014)  
   [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)  

2. Bahdanau, D., Cho, K., and Bengio, Y.  
   *Neural Machine Translation by Jointly Learning to Align and Translate* (2014)  
   [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)  

3. Vaswani, A., et al.  
   *Attention Is All You Need* (2017)  
   [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
   
4. Groeneveld, D. et al.  
   *OLMo: Accelerating the Science of Language Models* (2024)  
   [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)

5. Jiang, A. Q. et al.  
   *Mixtral of Experts* (2024)  
   [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
