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
        hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim
        latent_dim = config.output_dim[-1] if len(config.output_dim) > 1 else config.output_dim[-1]/len(config.cardinalities)
        dropout_p = config.dropout_p
        normalization = config.normalization
        activation = config.activation

        self.input_projections = nn.ModuleList([])
        for cardinality in self.cardinalities:
            if cardinality>1:   # Discrete Scenario
                if hidden_dims is None:                    # No hidden layers
                    self.input_projections.append(
                        EmbeddingLayer(
                            config=EmbeddingLayerConfig(
                                num_embeddings=cardinality+1,    # +1 for padding index
                                embedding_dim=latent_dim,
                                normalization=None,
                                activation=None,
                                dropout_p=dropout_p
                            )
                        )
                    )
                else:                                           # With hidden layers
                    layers = nn.ModuleList([])
                    layers.append(
                        EmbeddingLayer(
                            config=EmbeddingLayerConfig(
                                num_embeddings=cardinality+1,    # +1 for padding index
                                embedding_dim=hidden_dims[0],
                                normalization=None,
                                activation=activation,
                                dropout_p=dropout_p
                            )
                        )
                    )
                    for layer in range(1, len(hidden_dims)):
                        config = DenseLayerConfig(
                            input_dim=hidden_dims[layer-1],
                            output_dim=hidden_dims[layer],
                            normalization=normalization,
                            activation=activation,
                            dropout_p=dropout_p
                        )
                        layers.append(DenseLayer(config))
                    
                    layers.append(
                        DenseLayer(
                            config=DenseLayerConfig(
                                input_dim=hidden_dims[-1],
                                output_dim=latent_dim,
                                normalization=None,
                                activation=None,
                                dropout_p=dropout_p
                            )
                        )
                    )
                    self.input_projections.append(nn.Sequential(*layers))
            else:               # Continuous Scenario
                self.input_projections.append(
                    FeedForwardNetwork(
                        config=FeedForwardNetworkConfig(
                            input_dim=1,
                            hidden_dims=hidden_dims,
                            output_dim=latent_dim,
                            normalization=normalization,
                            activation=activation,
                            dropout_p=dropout_p
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

        embeddings = embeddings.view(embeddings.size(0), *self.output_dim)  # shape (batch_size, *output_dim)
                
        return embeddings, nan_mask # shape matches output_dim, (batch_size, num_attributes, 1)