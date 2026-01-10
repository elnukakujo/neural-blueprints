import torch

from .base import BaseInputProjection
from .....config.components.composite import FeedForwardNetworkConfig
from .....config.components.composite.projections.input import LinearInputProjectionConfig

import logging
logger = logging.getLogger(__name__)

class LinearInputProjection(BaseInputProjection):
    """
    Linear output projection.

    Args:
        config (LinearOutputProjectionConfig): Configuration for the linear output projection.
    """
    def __init__(
            self,
            config: LinearInputProjectionConfig
        ):
        super().__init__()
        from ... import FeedForwardNetwork
        self.input_dim = config.input_dim
        self.output_dim = [config.latent_dim]
        
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
            x (torch.Tensor): Input tensor of shape (batch_size, *input_dim).

        Returns:
            List[torch.Tensor]: List containing the output tensor of shape (batch_size, latent_dim).
        """

        x = torch.flatten(x, start_dim=1) # flatten the tensor

        return self.projection(x)  # shape: (batch_size, output_dim)