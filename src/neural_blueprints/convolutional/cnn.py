"""Convolutional Neural Network"""
import torch
import torch.nn as nn
from typing import List, Tuple
from ..base import NeuralNetworkBase


class ConvolutionalNN(NeuralNetworkBase):
    """
    Standard CNN with configurable convolutional and pooling layers.
    """
    
    def __init__(self, in_channels: int, num_classes: int, 
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 3, 3],
                 fc_dims: List[int] = [512, 256],
                 input_size: Tuple[int, int] = (28, 28)):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            conv_channels: Channels for each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            fc_dims: Dimensions of fully connected layers
            input_size: Input spatial dimensions (H, W)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        conv_layers = []
        prev_channels = in_channels
        
        for channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            prev_channels = channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate feature size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_size)
            conv_output = self.conv_layers(dummy_input)
            feature_size = conv_output.view(1, -1).size(1)
        
        # Fully connected layers
        fc_layers = []
        prev_dim = feature_size
        
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ])
            prev_dim = fc_dim
        
        fc_layers.append(nn.Linear(prev_dim, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification"""
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)