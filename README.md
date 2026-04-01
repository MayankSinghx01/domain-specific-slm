# Domain-Specific Small Language Model (SLM)

## Aim
To develop a **domain-specific Small Language Model (SLM)**, progressing towards a GPT-based Transformer architecture.

---

## Project Overview
This project focuses on building a **domain-specific Small Language Model (SLM)** from scratch and understanding sequence modeling through implementation.

The approach begins with attention-based encoder–decoder architectures using RNN/LSTM models and progresses towards implementing a **decoder-only Transformer (GPT-style)**.

The model is pretrained using language modeling on selective datasets (TinyStories/OpenWebText dataset or subsets of Wikipedia) to learn meaningful language representations.

Following pretraining, the project explores fine-tuning the model on domain-specific data using parameter-efficient techniques such as **LoRA** to overcome compute limitations.

Finally, the project aims to optimize inference through quantization (e.g., 16/8/4-bit) for memory-efficient deployment on compute-limited devices such as the Raspberry Pi.

As an optional extension, alignment techniques such as Reinforcement Learning from Human Feedback (RLHF) may be explored to further improve response quality.

---

## Current Progress

### Completed
- Implemented Next Word Predictor from scratch using RNN
- Implemented Next Word Predictor from scratch using LSTM
- Implemented attention-based encoder–decoder architecture using LSTM blocks

### In Progress
- Studying Transformer architecture and its components  
- Implementing encoder–decoder Transformer using self-attention  
- Implementing GPT-style decoder-only Transformer  

---

## Foundational Work (Brief)

Before focusing on language models, the following were implemented to build strong fundamentals:

- ANN from scratch using NumPy (MNIST classification)
- ANN using PyTorch
- CNN using PyTorch (Fashion MNIST)

These helped in understanding:
- Backpropagation  
- Gradient descent
- Regularisation Techniques
- Optimizers  
- Feature extraction

---

## Project Progress Tracker

## Phase 1: Sequence Models
- [x] RNN implementation  
- [x] LSTM implementation  
- [x] Next-word prediction  

## Phase 2: Attention
- [ ] Attention mechanism  
- [x] Encoder–Decoder architecture  

## Phase 3: Transformers
- [ ] Self-attention (scaled dot-product)  
- [ ] Multi-head attention  
- [ ] Positional encoding  
- [ ] Transformer block  

## Phase 4: GPT
- [ ] Decoder-only architecture  
- [ ] Masked self-attention  
- [ ] Pretraining  

## Phase 5: Optimization
- [ ] LoRA  
- [ ] Quantization  

## Phase 6: Deployment
- [ ] Edge deployment (Raspberry Pi)  

---

## Learning Roadmap

1. RNN → LSTM → Attention  
2. Encoder–Decoder Architectures  
3. Transformer Architecture  
4. GPT (Decoder-only Transformer)  
5. Fine-tuning (LoRA)  
6. Model Optimization (Quantization)  
7. Deployment on Edge Devices  

---

## Experiments (Ongoing)

| Model | Status | Notes |
|------|--------|------|
| RNN | Completed | Baseline sequence model |
| LSTM | Completed | Handles long-term dependencies better |
| RNN vs LSTM | Completed | Comparative analysis of sequence learning |
| Attention LSTM | Completed | Improved contextual learning |
| Transformer | In Progress | Core focus of current phase |
| GPT-style Model | In Progress | Final objective |

---

## Tech Stack
- Python  
- PyTorch  
- NumPy  

---

## References

1. Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le.  
   *Sequence to Sequence Learning with Neural Networks* (2014)  
   https://arxiv.org/abs/1409.3215  

2. Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.  
   *Neural Machine Translation by Jointly Learning to Align and Translate* (2014)  
   https://arxiv.org/abs/1409.0473  

3. Vaswani, Ashish, et al.  
   *Attention Is All You Need* (2017)  
   https://arxiv.org/abs/1706.03762  
