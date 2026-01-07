import torch
import torch.nn as nn

from .base import BaseOutputProjection
from .....config.components.composite import FeedForwardNetworkConfig
from .....config.components.composite.projections.output import LinearOutputProjectionConfig

import logging
logger = logging.getLogger(__name__)

class LinearOutputProjection(BaseOutputProjection):
    """
    Linear output projection.

    Args:
        config (LinearOutputProjectionConfig): Configuration for the linear output projection.
    """
    def __init__(
            self,
            config: LinearOutputProjectionConfig
        ):
        super().__init__()
        from ... import FeedForwardNetwork
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        
        hidden_dims = config.hidden_dims
        activation = config.activation
        normalization = config.normalization
        dropout_p = config.dropout_p
        
        self.projection = FeedForwardNetwork(
            FeedForwardNetworkConfig(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                output_dim=self.output_dim,
                activation=activation,
                normalization=normalization,
                dropout_p=dropout_p
            )
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through the output projection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).

        Returns:
            List of tensors, each of shape (batch_size, cardinality_i) for each attribute.
        """
        if len(x.size()) > 2:
            x = torch.flatten(x, start_dim=1) # shape: (batch_size, input_dim)

        return self.projection(x)  # shape: (batch_size, output_dim)