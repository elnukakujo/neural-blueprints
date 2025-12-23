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
        config (TabularOutputProjectionConfig): Configuration for the tabular output projection.
    """
    def __init__(
            self,
            config: TabularOutputProjectionConfig
        ):
        super().__init__()
        input_cardinalities = config.input_cardinalities
        output_cardinalities = config.output_cardinalities
        input_dim = config.input_dim
        hidden_dims = config.hidden_dims
        activation = config.activation
        normalization = config.normalization
        dropout_p = config.dropout_p
        if output_cardinalities is None or output_cardinalities == input_cardinalities:
            self.cardinalities = input_cardinalities
            self.with_same_cardinalities = True
            self.latent_dim = input_dim[-1] if len(input_dim)>1 else int(input_dim[-1] / len(input_cardinalities))
        else:
            self.cardinalities = output_cardinalities
            self.with_same_cardinalities = False
            self.latent_dim = torch.prod(torch.tensor(input_dim)).item()
            print(self.latent_dim)
        

        # Need to add case if input_cardinalities different from output_cardinalities for multi classification/regression scenarios
        self.output_projections = nn.ModuleList([])
        for cardinality in self.cardinalities:
            self.output_projections.append(
                FeedForwardNetwork(
                    config=FeedForwardNetworkConfig(
                        input_dim=self.latent_dim,
                        hidden_dims=hidden_dims,
                        output_dim=cardinality + 1 if cardinality > 1 else cardinality,
                        normalization=normalization,
                        activation=activation,
                        dropout_p=dropout_p,
                        final_activation="softmax" if cardinality > 1 else "sigmoid"
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
        if self.with_same_cardinalities:
            x = x.view(x.size(0), -1, self.latent_dim)       # shape: (batch_size, cardinalities, self.latent_dim)
        else:
            x = x.view(x.size(0), -1)                        # shape: (batch_size, latent_dim)

        predictions = []
        for i, layer in enumerate(self.output_projections):
            if self.with_same_cardinalities:
                col_data = x[:, i, :]  # shape: (batch_size, self.latent_dim)
            else:
                col_data = x  # shape: (batch_size, latent_dim)
            predictions.append(layer(col_data))
        if len(predictions) == 1:
            return predictions[0]
        return predictions