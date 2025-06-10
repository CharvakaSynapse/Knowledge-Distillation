# Knowledge Distillation on CIFAR-10: Deep Teacher, Lightweight Student

This repository implements and benchmarks multiple knowledge distillation (KD) techniques for training compact student models using a deeper teacher network on the CIFAR-10 dataset.

Inspired by:
- [Hinton et al., "Distilling the Knowledge in a Neural Network"](https://arxiv.org/pdf/1503.02531)
- [PyTorch Official KD Tutorial](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)

## Overview

We train a deep CNN as the **teacher**, and a lightweight CNN as the **student**, then apply various distillation techniques to improve the student’s performance without increasing its parameter count.

### Architectures:
- **DeepNN** (Teacher): ~1.19M parameters, 83.84% accuracy
- **LightNN** (Student): ~268K parameters, 72.35% accuracy (vanilla CE)

### Distillation Methods Implemented:

| Method                                | Description                                                       | Accuracy |
|--------------------------------------|-------------------------------------------------------------------|----------|
| **CE (Baseline)**                    | Cross-entropy training only                                       | 72.35%   |
| **CE + KD (Logits Matching)**        | KL-divergence on softened logits from teacher                     | 74.54%   |
| **CE + Gradient Attention Transfer** | Match saliency maps from gradients of intermediate features       | 72.50%   |
| **CE + Squared Activation Maps**     | Match squared attention maps (feature activation magnitudes)      | 72.69%   |
| **CE + KD + Grad Attention**         | Combined KL + attention map transfer                              | 73.29%   |

## Key Components

- `DeepNN`: High-capacity convolutional network (teacher)
- `LightNN`: Lightweight convolutional network with fewer layers
- `train()`, `test()`: Standard supervised training and evaluation loops
- `train_knowledge_distillation_logits()`: Implements soft target distillation (KD)
- `compute_gradient_attention()`: Computes attention via gradient saliency
- `train_attention_transfer_grad()`: Gradient-based attention distillation
- `compute_squared_attention()`: Computes squared activation-based attention
- `train_attention_transfer()`: Feature map-based attention transfer with MSE
- `train_attention_transfer(...)`: Final version that combines logit, CE, and attention losses

## Results Summary

```text
Teacher Accuracy               : 83.84%
Student (Cross-Entropy only)  : 72.35%
Student + KD (Logits)         : 74.54%
Student + Grad Attention      : 72.50%
Student + Squared Attention   : 72.69%
Student + KD + Grad Attention : 73.29%
