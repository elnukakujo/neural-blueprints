# Neural Network Backbones Library

A comprehensive collection of neural network architectures organized by inheritance hierarchy.

## Structure

```
neural_networks/
├── base/                    # Base classes
├── feedforward/            # Dense/MLP networks
├── convolutional/          # CNN-based architectures
├── recurrent/              # RNN-based architectures
├── transformer/            # Transformer-based architectures
└── utils/                  # Training utilities
```

## Installation

```bash
pip install torch torchvision numpy
```

## Usage

```python
from neural_networks.feedforward import FeedForwardNN, AutoEncoder
from neural_networks.transformer import BERT, GPT, VisionTransformer

# Example: Create a feedforward network
model = FeedForwardNN(input_dim=784, hidden_dims=[512, 256], output_dim=10)
```