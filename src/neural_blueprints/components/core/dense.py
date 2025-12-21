import torch
import torch.nn as nn

from ...config.components.core import DenseLayerConfig, NormalizationLayerConfig, DropoutLayerConfig
from ...utils import get_activation

class DenseLayer(nn.Module):
    """A fully connected dense layer with optional activation.
    
    Args:
        config (DenseLayerConfig): Configuration for the dense layer.
    """
    def __init__(self, config: DenseLayerConfig):
        super(DenseLayer, self).__init__()
        from ..core import NormalizationLayer, DropoutLayer
        
        linear_layer = nn.Linear(config.input_dim, config.output_dim)
        normalization_layer = NormalizationLayer(
            config=NormalizationLayerConfig(
                norm_type=config.normalization,
                num_features=config.output_dim
            )
        )
        activation_layer = get_activation(config.activation)
        dropout_layer = DropoutLayer(
            config=DropoutLayerConfig(
                p=config.dropout_p
            )
        )
        self.layer = nn.Sequential(
            linear_layer,
            normalization_layer,
            activation_layer,
            dropout_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dense layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.layer(x)