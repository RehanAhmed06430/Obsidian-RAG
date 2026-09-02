---
title: Neural Networks
tags: [neural-networks, deep-learning, AI]
created: 2026-08-05
---

# Neural Networks

## What are Neural Networks?

Neural networks are computing systems inspired by biological neural networks in the human brain. They consist of layers of interconnected nodes (neurons) that process information.

See also: [[Machine Learning]] for broader context.

## Architecture

### Basic Components

1. **Input Layer**: Receives raw data features
2. **Hidden Layers**: Intermediate processing layers
3. **Output Layer**: Produces final predictions

### Neuron Computation
Each neuron performs:
```
output = activation_function(Σ(weight_i * x_i) + bias)
```

## Types of Neural Networks

### Feedforward Networks
- Information flows in one direction
- Simplest form of neural network
- Used for: Classification, Regression

### Convolutional Neural Networks (CNNs)
- Specialized for grid-like data (images)
- Uses convolutional filters for feature extraction
- Key layers: Conv2D, Pooling, Dense

### Recurrent Neural Networks (RNNs)
- Designed for sequential data
- Has memory of previous inputs
- Variants: LSTM, GRU (solve vanishing gradient problem)

### [[Transformer Architecture]]
- Self-attention mechanism
- Parallelizable (unlike RNNs)
- Foundation of modern NLP

## Training Neural Networks

### Backpropagation
The core algorithm for training:
1. Forward pass: compute output
2. Calculate loss
3. Backward pass: compute gradients
4. Update weights using gradient descent

### Optimizers
- **SGD**: Basic gradient descent with momentum
- **Adam**: Adaptive learning rates, most popular
- **AdaGrad**: Good for sparse data

### Activation Functions
- ReLU: `max(0, x)` — most common for hidden layers
- Sigmoid: outputs between 0 and 1 — for binary classification
- Softmax: converts outputs to probability distribution
- GELU: smooth approximation of ReLU, used in transformers

## Common Challenges

- **Vanishing gradients**: Deep networks hard to train (solved by [[Transformer Architecture|transformers]] and residual connections)
- **Overfitting**: Use dropout, batch normalization, data augmentation
- **Computational cost**: GPU/TPU acceleration needed

## Key Papers
- "Deep Learning" by LeCun, Bengio, Hinton (2015)
- "Attention Is All You Need" — see [[Transformer Architecture]]
