import torch
import torch.nn as nn

from neural_blueprints.components.composite.projections.linear import LinearProjection
from neural_blueprints.config.components.composite.projections.linear import LinearProjectionConfig

from .base import BaseProjection

from ....config.components.composite import FeedForwardNetworkConfig
from ....config.components.composite.projections import TabularProjectionConfig
from ....config.components.core import EmbeddingLayerConfig

import logging
logger = logging.getLogger(__name__)

class DiscreteProjection(BaseProjection):
    def __init__(
            self,
            cardinality: list[int],
            output_dim: list[int],
            hidden_dims: list[int] = None,
            activation: str = None,
            normalization: str = None,
            dropout_p: float = 0.0
        ):
            from .. import FeedForwardNetwork
            from ...composite import FeedForwardNetwork
            from ...core import EmbeddingLayer
            super().__init__()
            self.input_dim = cardinality
            self.output_dim = output_dim

            # Determine dimensions for embedding layer
            if hidden_dims and len(hidden_dims) > 0:
                embedding_dim = hidden_dims[0]
            else:
                embedding_dim = output_dim

            # Create embedding layer
            layers = [
                EmbeddingLayer(
                    config=EmbeddingLayerConfig(
                        num_embeddings=cardinality[0] + 1,  # +1 for padding index
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
                            input_dim=[embedding_dim],
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

class NumericalProjection(BaseProjection):
    def __init__(
            self,
            output_dim: list[int],
            hidden_dims: list[int] = None,
            activation: str = None,
            normalization: str = None,
            dropout_p: float = 0.0
        ):
            super().__init__()
            self.input_dim = [1]
            self.output_dim = output_dim

            self.projection = LinearProjection(
                LinearProjectionConfig(
                    input_dim=self.input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=self.output_dim,
                    normalization=normalization,
                    activation=activation,
                    dropout_p=dropout_p
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the input projection module.
        """
        x = x.unsqueeze(-1)                                 # shape (batch_size, 1)
        emb = self.projection(x)                            # shape (batch_size, output_dim)
        is_masked = (x == -1).squeeze(-1)                   # shape (batch_size)
        return emb, is_masked

class TabularProjection(BaseProjection):
    def __init__(self,
                 config: TabularProjectionConfig
                ):
        super().__init__()
        if config.input_cardinalities is not None:
            tabular_2_linear = True
            self.input_dim = [len(config.input_cardinalities)]
            self.cardinalities = config.input_cardinalities
        else:
            tabular_2_linear = False
            self.input_dim = config.input_dim

        if config.output_cardinalities is not None:
            linear_2_tabular = True
            self.output_dim = [len(config.output_cardinalities)]
            self.cardinalities = config.output_cardinalities
        else:
            linear_2_tabular = False
            self.output_dim = config.output_dim

        assert tabular_2_linear or linear_2_tabular, "Either input_cardinalities or output_cardinalities must be provided."
        assert not (tabular_2_linear and linear_2_tabular), "Both input_cardinalities and output_cardinalities cannot be provided simultaneously."

        hidden_dims = config.hidden_dims
        dropout_p = config.dropout_p
        normalization = config.normalization
        activation = config.activation

        self.projections = nn.ModuleList([])

        for cardinality in self.cardinalities:
            if tabular_2_linear:   # Input Projection
                if cardinality>1:   # Discrete Scenario
                    # Add to input projections (wrap in Sequential if multiple layers, otherwise just the embedding)
                    self.projections.append(
                        DiscreteProjection(
                            cardinality=[cardinality],
                            output_dim=self.output_dim,
                            hidden_dims=hidden_dims,
                            normalization=normalization,
                            activation=activation,
                            dropout_p=dropout_p
                        )
                    )
                else:               # Continuous Scenario
                    self.projections.append(
                        NumericalProjection(
                            output_dim=self.output_dim,
                            hidden_dims=hidden_dims,
                            normalization=normalization,
                            activation=activation,
                            dropout_p=dropout_p
                        )
                    )
            elif linear_2_tabular:                   # Output Projection
                self.projections.append(
                    LinearProjection(
                        LinearProjectionConfig(
                            input_dim=config.output_dim,
                            hidden_dims=hidden_dims,
                            output_dim=[cardinality],
                            normalization=normalization,
                            activation=activation,
                            dropout_p=dropout_p
                        )
                    )
                )
            else:
                raise ValueError("Invalid configuration for TabularProjection.")

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

        for i, layer in enumerate(self. projections):
            col_data = x[:, i]
            emb, is_masked = layer(col_data)
            
            embeddings.append(emb)
            nan_mask.append(is_masked)                          # shape (batch_size)
        embeddings = torch.stack(embeddings, dim=1)             # shape (batch_size, num_attributes, output_dim)
        nan_mask = torch.stack(nan_mask, dim=1).bool()          # shape (batch_size, num_attributes)

        embeddings = embeddings.view(embeddings.size(0), *self.output_dim)  # shape (batch_size, *output_dim)
                
        return embeddings, nan_mask # shape matches output_dim, (batch_size, num_attributes, 1)