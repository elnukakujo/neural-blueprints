import torch
import torch.nn as nn
from ...config import ConvLayerConfig
from ...utils import get_activation

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
        return self.conv(x)
    

class Conv1dLayer(BaseConvLayer):
    def __init__(
        self,
        config: ConvLayerConfig
    ):
        """
        1D convolutional layer with optional batchnorm and activation.

        Args:
            in_channels (int)
            out_channels (int)
            kernel_size (int)
            stride (int)
            padding (int or None): if None, uses "same" padding (floor((k-1)/2))
            dilation (int)
            groups (int)
            bias (bool)
            activation (str | None)
            batch_norm (bool)
        """
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
        return self.conv(x)
    
class Conv2dTransposeLayer(BaseConvLayer):
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
        return self.conv(x)
    

class Conv1dTransposeLayer(BaseConvLayer):
    def __init__(
        self,
        config: ConvLayerConfig
    ):
        """
        1D convolutional layer with optional batchnorm and activation.

        Args:
            in_channels (int)
            out_channels (int)
            kernel_size (int)
            stride (int)
            padding (int or None): if None, uses "same" padding (floor((k-1)/2))
            dilation (int)
            groups (int)
            bias (bool)
            activation (str | None)
            batch_norm (bool)
        """
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
        return self.conv(x)