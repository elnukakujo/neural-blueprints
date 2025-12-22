import torch
import torch.nn as nn

from ... import FeedForwardNetwork
from .....config.components.composite import FeedForwardNetworkConfig
from .....config.components.composite.projections.output import TabularOutputProjectionConfig

import logging
logger = logging.getLogger(__name__)

class TabularOutputProjection(nn.Module):
    """
    Output projection for tabular data.

    Args:
        cardinalities (list[int]): List of cardinalities for each categorical attribute.
        latent_dim (int): Dimension of the latent representation.
        hidden_dims (list[int]): List of hidden dimensions for the feedforward networks.
        activation (str): Activation function to use.
        normalization (str): Normalization method to use.
        dropout_p (float): Dropout probability.
    """
    def __init__(
            self,
            config: TabularOutputProjectionConfig
        ):
        super().__init__()
        cardinalities = config.cardinalities
        latent_dim = config.input_dim
        hidden_dims = config.hidden_dims
        activation = config.activation
        normalization = config.normalization
        dropout_p = config.dropout_p
        final_activation = config.final_activation

        self.output_projections = nn.ModuleList([])
        for cardinality in cardinalities:
            self.output_projections.append(
                FeedForwardNetwork(
                    config=FeedForwardNetworkConfig(
                        input_dim=latent_dim,
                        hidden_dims=hidden_dims,
                        output_dim=cardinality + 1 if cardinality > 1 else cardinality,
                        normalization=normalization,
                        activation=activation,
                        dropout_p=dropout_p,
                        final_activation=final_activation
                    )
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through the output projection.

        Args:
            x: Input tensor of shape (batch_size, seq_length, latent_dim).

        Returns:
            List of tensors, each of shape (batch_size, cardinality_i) for each attribute.
        """
        predictions = []
        for i, layer in enumerate(self.output_projections):
            col_data = x[:, i, :]  # shape: (batch_size, latent_dim)
            predictions.append(layer(col_data))
        return predictions