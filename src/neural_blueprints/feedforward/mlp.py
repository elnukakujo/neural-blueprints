"""Feedforward Neural Network (MLP)"""
import torch
import torch.nn as nn
from typing import List, Optional
from ..base import NeuralNetworkBase


class FeedForwardNN(NeuralNetworkBase):
    """
    Multi-Layer Perceptron (MLP) with configurable architecture.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, activation: str = "relu", 
                 dropout: float = 0.0, batch_norm: bool = False):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function ('relu', 'tanh', 'gelu')
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)