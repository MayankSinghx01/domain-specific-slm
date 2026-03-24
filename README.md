# Domain Specific SLM

## Aim

To develop a domain-specific Small Language Model (SLM), progressing towards a GPT-based Transformer architecture.

## Project Overview

This project focuses on building a domain-specific Small Language Model (SLM) from scratch and understand sequence modeling through implementation.

The approach begins with attention-based encoder–decoder architectures using RNN/LSTM models, and progresses towards implementing a decoder-only Transformer (GPT-style). The model is pretrained using language modeling on selective datasets (TinyStories/OpenWebText dataset or subsets of Wikipedia) to learn meaningful language representations.

Following pretraining, the project explores fine-tuning the model on domain-specific data using parameter-efficient techniques such as LoRA to overcome compute limitations.

Finally, the project aims to optimize inference through quantization (e.g., 16/8/4-bit) for memory-efficient deployment on compute limited devices such as the Raspberry Pi.

As an optional extension, alignment techniques such as Reinforcement Learning from Human Feedback (RLHF) may be explored to further improve response quality.

## Current Progress

- Implemented Next Word Predictor from scratch on both RNN/LSTM architectures.
- Implemented attention-based encoder-decoder architecture using LSTM blocks.
- Currently studying Transformer architecture and its components.
