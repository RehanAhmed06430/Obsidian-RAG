---
title: Transformer Architecture
tags: [transformers, NLP, deep-learning, attention]
created: 2026-08-10
---

# Transformer Architecture

## Introduction

The Transformer is a deep learning architecture introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017). It revolutionized NLP and is the foundation of models like GPT, BERT, and [[Machine Learning|modern ML systems]].

## Core Innovation: Self-Attention

Self-attention allows the model to weigh the importance of different parts of the input when processing each element.

### Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q (Query)**: What am I looking for?
- **K (Key)**: What do I contain?
- **V (Value)**: What information do I provide?
- **d_k**: Dimension of keys (scaling factor)

### Multi-Head Attention
- Run multiple attention operations in parallel
- Each head learns different relationship patterns
- Concatenate and project results
- Usually 8 or 16 heads in modern models

## Architecture Components

### Encoder
- Processes input sequence
- Self-attention + feedforward layers
- Each layer: Multi-head attention → Add & Norm → FFN → Add & Norm
- Used in BERT-style models

### Decoder
- Generates output sequence
- Masked self-attention (can't see future tokens)
- Cross-attention to encoder output
- Used in GPT-style models

### Positional Encoding
- Transformers don't have inherent sequence order
- Position information is added to embeddings
- Sinusoidal functions in original paper
- Learned embeddings in modern variants

## Key Concepts

### Residual Connections
- Skip connections around each sub-layer
- Helps with gradient flow in deep networks
- Enables training very deep transformers

### Layer Normalization
- Normalizes activations within each layer
- Stabilizes training
- Applied before or after sub-layers (Pre-LN vs Post-LN)

## Applications

### Natural Language Processing
- Machine Translation (original use case)
- Text Generation (GPT series)
- Text Classification (BERT)
- Question Answering

### Computer Vision
- Vision Transformer (ViT)
- Image generation (DALL-E)

### Multimodal
- CLIP: Image-text understanding
- GPT-4V: Vision + Language

## Modern Variants
- **GPT**: Decoder-only, autoregressive generation
- **BERT**: Encoder-only, bidirectional understanding
- **T5**: Encoder-decoder, text-to-text framework
- **LLaMA**: Efficient decoder-only, open source

## Related Notes
- [[Neural Networks]] — Underlying neural network concepts
- [[Machine Learning]] — Broader ML context
- [[Python Tips]] — Implementation tips
