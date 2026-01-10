import torch
import torch.nn as nn

from .base import BaseOutputProjection

from .....config.components.core import DenseLayerConfig
from .....config.components.composite import FeedForwardNetworkConfig
from .....config.components.composite.projections.output import TabularOutputProjectionConfig

import logging
logger = logging.getLogger(__name__)

class TabularOutputProjection(BaseOutputProjection):
    """
    Output projection for tabular data.

    Args:
        config (TabularOutputProjectionConfig): Configuration for the tabular output projection.
    """
    def __init__(
            self,
            config: TabularOutputProjectionConfig
        ):
        super().__init__()
        from .....components.core import DenseLayer
        from ... import FeedForwardNetwork
        self.input_dim = config.input_dim
        self.output_dim = tuple([cardinality + 1] if cardinality > 1 else [cardinality] for cardinality in config.cardinalities)

        self.cardinalities = config.cardinalities

        latent_dim = self.input_dim[-1]
        hidden_dims = config.hidden_dims
        activation = config.activation
        normalization = config.normalization
        dropout_p = config.dropout_p

        # Need to add case if input_cardinalities different from cardinalities for multi classification/regression scenarios
        self.output_projections = nn.ModuleList([])
        for cardinality in self.cardinalities:
            if hidden_dims is None or len(hidden_dims) == 0:
                self.output_projections.append(
                    DenseLayer(
                        config=DenseLayerConfig(
                            input_dim=[latent_dim],
                            output_dim=[cardinality + 1] if cardinality > 1 else [cardinality],
                            normalization=None,
                            activation="sigmoid",
                            dropout_p=0,
                        )
                    )
                )
            else:
                self.output_projections.append(
                    FeedForwardNetwork(
                        config=FeedForwardNetworkConfig(
                            input_dim=[latent_dim],
                            hidden_dims=hidden_dims,
                            output_dim=[cardinality + 1] if cardinality > 1 else [cardinality],
                            normalization=normalization,
                            activation=activation,
                            dropout_p=dropout_p,
                            final_activation="sigmoid"
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
            predictions.append(layer(x[:, i, :]))  # shape: (batch_size, cardinality_i)
        if len(predictions) == 1:
            return predictions[0]
        return predictions