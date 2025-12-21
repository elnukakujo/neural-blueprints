import torch
import torch.nn as nn

from ...config.components.core import NormalizationLayerConfig

import logging
logger = logging.getLogger(__name__)

class NormalizationLayer(nn.Module):
    """A layer that applies normalization to its input tensor.
    
    Args:
        config (NormalizationConfig): Configuration for the normalization layer.
    """
    def __init__(self, config: NormalizationLayerConfig):
        super(NormalizationLayer, self).__init__()
        if config is None or config.norm_type is None:
            self.norm_type = None
            self.network = nn.Identity()
            return

        self.norm_type = config.norm_type.lower()
        if "batchnorm" in self.norm_type:
            if "1d" in self.norm_type:
                self.network = nn.BatchNorm1d(config.num_features)
            elif "2d" in self.norm_type:
                self.network = nn.BatchNorm2d(config.num_features)
            elif "3d" in self.norm_type:
                self.network = nn.BatchNorm3d(config.num_features)
            else:
                raise ValueError(f"Unsupported batchnorm type: {self.norm_type}. Supported types: 'batchnorm1d', 'batchnorm2d', 'batchnorm3d'")
        elif "layernorm" in self.norm_type:
            self.network = nn.LayerNorm(config.num_features)
        elif "instancenorm" in self.norm_type:
            if "1d" in self.norm_type:
                self.network = nn.InstanceNorm1d(config.num_features)
            elif "2d" in self.norm_type:
                self.network = nn.InstanceNorm2d(config.num_features)
            elif "3d" in self.norm_type:
                self.network = nn.InstanceNorm3d(config.num_features)
            else:
                raise ValueError(f"Unsupported instancenorm type: {self.norm_type}. Supported types: 'instancenorm1d', 'instancenorm2d', 'instancenorm3d'")
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}. Supported types: 'batchnorm1d', 'batchnorm2d', 'layernorm'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the reshape layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
        
        Returns:
            Normalized tensor of the same shape as input.
        """
        try:
            if self.norm_type is not None and "batchnorm" in self.norm_type and x.dim() > 2:
                # For batchnorm, we need to flatten the input except for the batch dimension
                x = x.transpose(1, 2) 
                x = self.network(x)
                x = x.transpose(1, 2)
                return x
            else:
                return self.network(x)
        except Exception as e:
            logger.debug(f"Input tensor shape: {x.shape}")
            logger.debug(f"Normalization layer: {self.network}")
            raise RuntimeError(f"Error during normalization forward pass: {e}")