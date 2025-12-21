import torch
import torch.nn as nn
from abc import abstractmethod

from ...config.components.core import ConvLayerConfig, DropoutLayerConfig
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
        self.activation = config.activation
        self.dropout_p = config.dropout_p

class Conv3dLayer(BaseConvLayer):
    """A 3D convolutional layer with optional batchnorm and activation.
    
    Args:
        config (ConvLayerConfig): Configuration for the convolutional layer.
    """
    def __init__(
        self,
        config: ConvLayerConfig
    ):
        super().__init__(config)
        from ..core import DropoutLayer

        if self.padding is None:
            self.padding = (self.kernel_size - 1) // 2

        layers = [
            nn.Conv3d(
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

        layers.append(get_activation(self.activation))
        layers.append(DropoutLayer(
            config=DropoutLayerConfig(
                p=self.dropout_p
            )
        ))

        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2D convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.layer(x)

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
        from ..core import DropoutLayer
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

        layers.append(get_activation(self.activation))
        layers.append(DropoutLayer(
            config=DropoutLayerConfig(
                p=self.dropout_p
            )
        ))

        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2D convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.layer(x)
    

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
        from ..core import DropoutLayer
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

        layers.append(get_activation(self.activation))
        layers.append(DropoutLayer(
            config=DropoutLayerConfig(
                p=self.dropout_p
            )
        ))
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 1D convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            Output tensor of shape (batch_size, out_channels, length).
        """
        return self.layer(x)
    
class Conv3dTransposeLayer(BaseConvLayer):
    """A 3D transposed convolutional layer with optional batchnorm and activation.
    
    Args:
        config (ConvLayerConfig): Configuration for the convolutional layer.
    """
    def __init__(
        self,
        config: ConvLayerConfig
    ):
        super().__init__(config)
        from ..core import DropoutLayer

        if self.padding is None:
            self.padding = (self.kernel_size - 1) // 2

        layers = [
            nn.ConvTranspose3d(
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

        layers.append(get_activation(self.activation))
        layers.append(DropoutLayer(
            config=DropoutLayerConfig(
                p=self.dropout_p
            )
        ))

        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3D transposed convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
            
        Returns:
            Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        return self.layer(x)
    
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
        from ..core import DropoutLayer
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

        layers.append(get_activation(self.activation))
        layers.append(DropoutLayer(
            config=DropoutLayerConfig(
                p=self.dropout_p
            )
        ))

        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2D transposed convolutional layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.layer(x)
    

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
        from ..core import DropoutLayer
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

        layers.append(get_activation(self.activation))
        layers.append(DropoutLayer(
            config=DropoutLayerConfig(
                p=self.dropout_p
            )
        ))
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 1D transposed convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
        
        Returns:
            Output tensor of shape (batch_size, out_channels, length).
        """
        return self.layer(x)