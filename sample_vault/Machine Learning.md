---
title: Machine Learning
tags: [machine-learning, AI, fundamentals]
created: 2026-08-01
---

# Machine Learning

## Overview

Machine Learning is a subset of **Artificial Intelligence** that focuses on building systems that learn from data. Instead of being explicitly programmed, these systems improve their performance through experience.

## Types of Machine Learning

### Supervised Learning
- Uses labeled training data
- Common algorithms: [[Neural Networks|Neural Networks]], SVM, Random Forest
- Applications: Classification, Regression

### Unsupervised Learning
- Works with unlabeled data
- Common algorithms: K-Means, DBSCAN, PCA
- Applications: Clustering, Dimensionality Reduction

### Reinforcement Learning
- Agent learns through interaction with environment
- Uses reward signals to guide learning
- Applications: Game playing, Robotics

## Key Concepts

### Bias-Variance Tradeoff
The fundamental tension in machine learning:
- **Bias**: Error from oversimplifying the model (underfitting)
- **Variance**: Error from overcomplexity (overfitting)
- The goal is finding the sweet spot between them

### Overfitting
When a model learns the training data too well, including noise:
- High training accuracy, low test accuracy
- Solution: Regularization, more data, simpler model
- Cross-validation helps detect overfitting

### Feature Engineering
Transforming raw data into meaningful features:
- Normalization and standardization
- One-hot encoding for categorical variables
- Feature selection and extraction
- Domain knowledge is crucial

## Tools and Frameworks

- **scikit-learn**: Classic ML algorithms
- [[Python Tips|Python for ML]]: NumPy, Pandas, Matplotlib
- TensorFlow and PyTorch for [[Neural Networks|deep learning]]

## Related Notes
- [[Neural Networks]] — Deep learning foundation
- [[Transformer Architecture]] — Modern architecture revolution
- [[Python Tips]] — Useful Python techniques
