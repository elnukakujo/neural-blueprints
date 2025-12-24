import torch
import torch.nn as nn

from ...config.components.core import PoolingLayerConfig

class Pooling3dLayer(nn.Module):
    """A 3D pooling layer that supports different pooling types.

    Args:
        config (PoolingLayerConfig): Configuration for the pooling layer.
    """
    def __init__(self, config: PoolingLayerConfig):
        super(Pooling3dLayer, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=config.kernel_size, stride=config.stride) if config.pool_type.lower() == 'max' else nn.AvgPool3d(kernel_size=config.kernel_size, stride=config.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3d pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            Output tensor after pooling.
        """
        return self.pool(x)

class Pooling2dLayer(nn.Module):
    """A 2D pooling layer that supports different pooling types.
    
    Args:
        config (PoolingLayerConfig): Configuration for the pooling layer.
    """
    def __init__(self, config: PoolingLayerConfig):
        super(Pooling2dLayer, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=config.kernel_size, stride=config.stride) if config.pool_type == 'max' else nn.AvgPool2d(kernel_size=config.kernel_size, stride=config.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2d pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output tensor after pooling.
        """
        return self.pool(x)
    
class Pooling1dLayer(nn.Module):
    """A 1D pooling layer that supports different pooling types.

    Args:
        config (PoolingLayerConfig): Configuration for the pooling layer.
    """
    def __init__(self, config: PoolingLayerConfig):
        super(Pooling1dLayer, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=config.kernel_size, stride=config.stride) if config.pool_type.lower() == 'max' else nn.AvgPool1d(kernel_size=config.kernel_size, stride=config.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 1d pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            Output tensor after pooling.
        """
        return self.pool(x)