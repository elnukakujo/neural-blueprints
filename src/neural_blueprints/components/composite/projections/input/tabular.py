import torch
import torch.nn as nn

from ... import FeedForwardNetwork
from ....core import EmbeddingLayer, DenseLayer
from .....config.components.composite import FeedForwardNetworkConfig
from .....config.components.composite.projections.input import TabularInputProjectionConfig
from .....config.components.core import EmbeddingLayerConfig, DenseLayerConfig

import logging
logger = logging.getLogger(__name__)

class TabularInputProjection(nn.Module):
    """
        Mixed type Tabular Data Input Projection Module.
        Projects each attribute (discrete or continuous) into a latent space of given dimensionality.

        Args:
            config (TabularInputProjectionConfig): Configuration for the Tabular Input Projection.
    """
    def __init__(self,
                 config: TabularInputProjectionConfig
                ):
        super().__init__()
        self.cardinalities = config.cardinalities
        self.hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim
        self.dropout_p = config.dropout_p
        self.normalization = config.normalization
        self.activation = config.activation

        self.input_projections = nn.ModuleList([])
        for cardinality in self.cardinalities:
            if cardinality>1:   # Discrete Scenario
                if self.hidden_dims is None:                    # No hidden layers
                    self.input_projections.append(
                        EmbeddingLayer(
                            config=EmbeddingLayerConfig(
                                num_embeddings=cardinality+1,   # +1 for padding idx
                                embedding_dim=self.output_dim,
                                normalization=None,
                                activation=None
                            )
                        )
                    )
                else:                                           # With hidden layers
                    layers = nn.ModuleList([])
                    layers.append(
                        EmbeddingLayer(
                            config=EmbeddingLayerConfig(
                                num_embeddings=cardinality+1,   # +1 for padding idx
                                embedding_dim=self.hidden_dims[0],
                                normalization=None,
                                activation=self.activation
                            )
                        )
                    )
                    for layer in range(1, len(self.hidden_dims)):
                        config = DenseLayerConfig(
                            input_dim=self.hidden_dims[layer-1],
                            output_dim=self.hidden_dims[layer],
                            normalization=self.normalization,
                            activation=self.activation
                        )
                        layers.append(DenseLayer(config))
                    
                    layers.append(
                        DenseLayer(
                            config=DenseLayerConfig(
                                input_dim=self.hidden_dims[-1],
                                output_dim=self.output_dim,
                                normalization=None,
                                activation=None
                            )
                        )
                    )
                    self.input_projections.append(nn.Sequential(*layers))
            else:               # Continuous Scenario
                self.input_projections.append(
                    FeedForwardNetwork(
                        config=FeedForwardNetworkConfig(
                            input_dim=1,
                            hidden_dims=self.hidden_dims,
                            output_dim=self.output_dim,
                            normalization=self.normalization,
                            activation=self.activation
                        )
                    )
                )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the input projection module.

        Args:
            x: Input tensor of shape (batch_size, num_attributes).

        Returns:
            Projected tensor of shape (batch_size, num_attributes, output_dim).
            Mask tensor indicating NaN values of shape (batch_size, num_attributes).
        """

        embeddings = []
        nan_mask = []

        for i, layer in enumerate(self.input_projections):
            if self.cardinalities[i]>1:
                col_data = x[:, i]                              # shape (batch_size)
                emb = layer(col_data)                           # shape (batch_size, output_dim)
                is_masked = (col_data == 0)                     # shape (batch_size)
            else:
                col_data = x[:, i].unsqueeze(1)                 # shape (batch_size, 1)
                emb = layer(col_data)                           # shape (batch_size, output_dim)
                is_masked = (col_data == -1).squeeze(1)         # shape (batch_size)
            assert not torch.isnan(emb).any(), f"NaN values found in InputProjection output for column {i}"
            embeddings.append(emb)
            nan_mask.append(is_masked)                          # shape (batch_size)
        embeddings = torch.stack(embeddings, dim=1)             # shape (batch_size, num_attributes, output_dim)
        nan_mask = torch.stack(nan_mask, dim=1).bool()          # shape (batch_size, num_attributes)
                
        return embeddings, nan_mask # shape (batch_size, num_attributes, output_dim), (batch_size, num_attributes, 1)