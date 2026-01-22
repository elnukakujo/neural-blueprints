import torch

from .base import BaseProjection
from ....config.components.composite import FeedForwardNetworkConfig
from ....config.components.composite.projections import LinearProjectionConfig

import logging
logger = logging.getLogger(__name__)

class LinearProjection(BaseProjection):
    def __init__(
            self,
            config: LinearProjectionConfig
        ):
        super().__init__()
        from .. import FeedForwardNetwork
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        
        hidden_dims = config.hidden_dims
        activation = config.activation
        normalization = config.normalization
        dropout_p = config.dropout_p
        final_activation = config.final_activation
        
        self.projection = FeedForwardNetwork(
            FeedForwardNetworkConfig(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                output_dim=self.output_dim,
                activation=activation,
                normalization=normalization,
                dropout_p=dropout_p,
                final_activation=final_activation
            )
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.projection(x)  # shape: (batch_size, output_dim)