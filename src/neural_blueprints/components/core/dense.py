import torch
import torch.nn as nn
from ...config import DenseLayerConfig, NormalizationConfig
from ...utils import get_activation, get_normalization

class DenseLayer(nn.Module):
    """A fully connected dense layer with optional activation.
    
    Args:
        config (DenseLayerConfig): Configuration for the dense layer.
    """
    def __init__(self, config: DenseLayerConfig):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(config.input_dim, config.output_dim)
        norm_config = NormalizationConfig(
            norm_type=config.normalization,
            num_features=config.output_dim
        )
        self.normalization = get_normalization(norm_config)
        self.activation = get_activation(config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dense layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        x = self.linear(x.float())
        x = self.normalization(x)
        x = self.activation(x)
        return x