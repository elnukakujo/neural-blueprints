import torch
import torch.nn as nn

from .base import BaseInputProjection

from .....config.components.composite import FeedForwardNetworkConfig
from .....config.components.composite.projections.input import TabularInputProjectionConfig
from .....config.components.core import EmbeddingLayerConfig

import logging
logger = logging.getLogger(__name__)

class DiscreteProjection(BaseInputProjection):
    def __init__(
            self,
            cardinality: int,
            output_dim: int,
            hidden_dims: list[int] = None,
            activation: str = None,
            normalization: str = None,
            dropout_p: float = 0.0
        ):
            from ... import FeedForwardNetwork
            from ....composite import FeedForwardNetwork
            from ....core import EmbeddingLayer
            super().__init__()
            self.input_dim = cardinality
            self.output_dim = output_dim

            # Determine dimensions for embedding layer
            if hidden_dims and len(hidden_dims) > 0:
                embedding_dim = hidden_dims[0]
                hidden_dims = hidden_dims[1:]
            else:
                embedding_dim = output_dim

            # Create embedding layer
            layers = [
                EmbeddingLayer(
                    config=EmbeddingLayerConfig(
                        num_embeddings=cardinality + 1,  # +1 for padding index
                        embedding_dim=embedding_dim,
                        normalization=normalization,
                        activation=activation,
                        dropout_p=dropout_p
                    )
                )
            ]

            # Add hidden layers if specified
            if hidden_dims and len(hidden_dims) > 0:
                layers.append(
                    FeedForwardNetwork(
                        config=FeedForwardNetworkConfig(
                            input_dim=embedding_dim,
                            hidden_dims=hidden_dims,
                            output_dim=output_dim,
                            normalization=normalization,
                            activation=activation,
                            dropout_p=dropout_p,
                            final_activation=activation
                        )
                    )
                )
            self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the input projection module.
        """
        emb = self.projection(x)                            # shape (batch_size, output_dim)
        is_masked = (x == 0)                                # shape (batch_size)
        return emb, is_masked

class NumericalProjection(BaseInputProjection):
    def __init__(
            self,
            output_dim: int,
            hidden_dims: list[int] = [],
            activation: str = None,
            normalization: str = None,
            dropout_p: float = 0.0
        ):
            from ... import FeedForwardNetwork
            super().__init__()
            self.input_dim = [1]
            self.output_dim = [output_dim]

            self.projection = FeedForwardNetwork(
                config=FeedForwardNetworkConfig(
                    input_dim=1,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                    normalization=normalization,
                    activation=activation,
                    dropout_p=dropout_p,
                    final_activation=activation
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the input projection module.
        """
        emb = self.projection(x)                # shape (batch_size, output_dim)
        is_masked = (x == -1).squeeze(1)        # shape (batch_size)
        return emb, is_masked

class TabularInputProjection(BaseInputProjection):
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
        self.input_dim = [len(config.cardinalities)]
        self.output_dim = [self.input_dim[0], config.latent_dim]

        self.cardinalities = config.cardinalities
        hidden_dims = config.hidden_dims
        latent_dim = config.latent_dim
        dropout_p = config.dropout_p
        normalization = config.normalization
        activation = config.activation

        self.input_projections = nn.ModuleList([])

        for cardinality in self.cardinalities:
            if cardinality>1:   # Discrete Scenario
                # Add to input projections (wrap in Sequential if multiple layers, otherwise just the embedding)
                self.input_projections.append(
                    DiscreteProjection(
                        cardinality=cardinality,
                        output_dim=latent_dim,
                        hidden_dims=hidden_dims,
                        normalization=normalization,
                        activation=activation,
                        dropout_p=dropout_p
                    )
                )
            else:               # Continuous Scenario
                self.input_projections.append(
                    NumericalProjection(
                        output_dim=latent_dim,
                        hidden_dims=hidden_dims,
                        normalization=normalization,
                        activation=activation,
                        dropout_p=dropout_p
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
            col_data = x[:, i]
            emb, is_masked = layer(col_data)
            
            embeddings.append(emb)
            nan_mask.append(is_masked)                          # shape (batch_size)
        embeddings = torch.stack(embeddings, dim=1)             # shape (batch_size, num_attributes, output_dim)
        nan_mask = torch.stack(nan_mask, dim=1).bool()          # shape (batch_size, num_attributes)

        embeddings = embeddings.view(embeddings.size(0), *self.output_dim)  # shape (batch_size, *output_dim)
                
        return embeddings, nan_mask # shape matches output_dim, (batch_size, num_attributes, 1)