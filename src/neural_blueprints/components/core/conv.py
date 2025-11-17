import torch
import torch.nn as nn
from abc import abstractmethod

from ...config import ConvLayerConfig
from ...utils import get_activation


@abstractmethod
class BaseConvLayer(nn.Module):
    def __init__(self, config: ConvLayerConfig):
        super(BaseConvLayer, self).__init__()
        
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.padding = config.padding
        self.output_padding = config.output_padding
        self.dilation = config.dilation
        self.groups = config.groups
        self.bias = config.bias
        self.batch_norm = config.batch_norm
        self.activation = config.activation

class Conv2dLayer(BaseConvLayer):
    """A 2D convolutional layer with optional batchnorm and activation.
    
    Args:
        config (ConvLayerConfig): Configuration for the convolutional layer.
    """
    def __init__(
        self,
        config: ConvLayerConfig
    ):
        super().__init__(config)
        if self.padding is None:
            self.padding = (self.kernel_size - 1) // 2

        layers = [
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
        ]

        layers.append(nn.BatchNorm2d(self.out_channels)) if self.batch_norm else None
        layers.append(get_activation(self.activation))
        

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2D convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.conv(x)
    

class Conv1dLayer(BaseConvLayer):
    """A 1D convolutional layer with optional batchnorm and activation.
    
    Args:
        config (ConvLayerConfig): Configuration for the convolutional layer.
    """
    def __init__(
        self,
        config: ConvLayerConfig
    ):
        super().__init__(config)
        if self.padding is None:
            self.padding = (self.kernel_size - 1) // 2

        layers = [
            nn.Conv1d(
                self.in_channels, 
                self.out_channels, 
                self.kernel_size,
                stride=self.stride, 
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias
            )
        ]

        layers.append(nn.BatchNorm1d(self.out_channels)) if self.batch_norm else None
        layers.append(get_activation(self.activation))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 1D convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            Output tensor of shape (batch_size, out_channels, length).
        """
        return self.conv(x)
    
class Conv2dTransposeLayer(BaseConvLayer):
    """A 2D transposed convolutional layer with optional batchnorm and activation.
    
    Args:
        config (ConvLayerConfig): Configuration for the convolutional layer.
    """
    def __init__(
        self,
        config: ConvLayerConfig
    ):
        super().__init__(config)
        if self.padding is None:
            self.padding = (self.kernel_size - 1) // 2

        layers = [
            nn.ConvTranspose2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
        ]

        layers.append(nn.BatchNorm2d(self.out_channels)) if self.batch_norm else None
        layers.append(get_activation(self.activation))
        

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2D transposed convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.conv(x)
    

class Conv1dTransposeLayer(BaseConvLayer):
    """A 1D transposed convolutional layer with optional batchnorm and activation.
    
    Args:
        config (ConvLayerConfig): Configuration for the convolutional layer.
    """
    def __init__(
        self,
        config: ConvLayerConfig
    ):
        super().__init__(config)
        if self.padding is None:
            self.padding = (self.kernel_size - 1) // 2

        layers = [
            nn.ConvTranspose1d(
                self.in_channels, 
                self.out_channels, 
                self.kernel_size,
                stride=self.stride, 
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias
            )
        ]

        layers.append(nn.BatchNorm1d(self.out_channels)) if self.batch_norm else None
        layers.append(get_activation(self.activation))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 1D transposed convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            Output tensor of shape (batch_size, out_channels, length).
        """
        return self.conv(x)