"""U-Net Architecture"""
import torch
import torch.nn as nn
from typing import List
from ..base import NeuralNetworkBase


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(NeuralNetworkBase):
    """
    U-Net architecture for image segmentation and generation tasks.
    Often used as backbone for diffusion models.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 features: List[int] = [64, 128, 256, 512]):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Feature dimensions at each level
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        prev_channels = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(prev_channels, feature))
            prev_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections"""
        skip_connections = []
        
        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for idx in range(len(self.decoder)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx](x)
        
        return self.final_conv(x)