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
            self.output_dim = [output_dim]

            # Determine dimensions for embedding layer
            if hidden_dims and len(hidden_dims) > 0:
                embedding_dim = hidden_dims[0]
            else:
                embedding_dim = output_dim[0]

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

class TabularOutputProjection(BaseProjection):
    def __init__(self,
                 config: TabularProjectionConfig
                ):
        super().__init__()
        self.input_dim = config.input_dim
        projection_dim = self.input_dim[-1]
        self.output_dim = [len(config.output_cardinalities)]
        self.cardinalities = config.output_cardinalities

        hidden_dims = config.hidden_dims
        dropout_p = config.dropout_p
        normalization = config.normalization
        activation = config.activation

        self.projections = nn.ModuleList([])

        for cardinality in self.cardinalities:
            self.projections.append(
                LinearProjection(
                    LinearProjectionConfig(
                        input_dim=[projection_dim],
                        hidden_dims=hidden_dims,
                        output_dim=[cardinality + 1 if cardinality > 1 else 1],
                        normalization=normalization,
                        activation=activation,
                        dropout_p=dropout_p
                    )
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the output projection module.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Projected list of shape num_attributes*(batch_size, cardinality).
        """

        outputs = []

        for i, layer in enumerate(self.projections):
            if x.dim() > 2:
                col_data = x[:, i, :]                           # shape (batch_size, projection_dim)
            else:
                col_data = x
            emb = layer(col_data)                               # shape (batch_size, cardinality)
            outputs.append(emb)
                
        return outputs                                      # shape num_attributes*(batch_size, cardinality)

class TabularInputProjection(BaseProjection):
    def __init__(self,
                 config: TabularProjectionConfig
                ):
        super().__init__()
        self.input_dim = [len(config.input_cardinalities)]
        projection_dim = config.output_dim
        self.output_dim = [len(config.input_cardinalities), projection_dim[0]]
        
        self.cardinalities = config.input_cardinalities

        hidden_dims = config.hidden_dims
        dropout_p = config.dropout_p
        normalization = config.normalization
        activation = config.activation

        self.projections = nn.ModuleList([])

        for cardinality in self.cardinalities:
            if cardinality>1:   # Discrete Scenario
                # Add to input projections (wrap in Sequential if multiple layers, otherwise just the embedding)
                self.projections.append(
                    DiscreteProjection(
                        cardinality=[cardinality],
                        output_dim=projection_dim,
                        hidden_dims=hidden_dims,
                        normalization=normalization,
                        activation=activation,
                        dropout_p=dropout_p
                    )
                )
            else:               # Continuous Scenario
                self.projections.append(
                    NumericalProjection(
                        output_dim=projection_dim,
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

        for i, layer in enumerate(self.projections):
            col_data = x[:, i]
            emb, is_masked = layer(col_data)
            
            embeddings.append(emb)
            nan_mask.append(is_masked)                          # shape (batch_size)
        embeddings = torch.stack(embeddings, dim=1)             # shape (batch_size, num_attributes, output_dim)
        nan_mask = torch.stack(nan_mask, dim=1).bool()          # shape (batch_size, num_attributes)

        embeddings = embeddings.view(embeddings.size(0), *self.output_dim)  # shape (batch_size, *output_dim)
                
        return embeddings # shape matches output_dim, (batch_size, num_attributes, 1)
    
class TabularProjection(BaseProjection):
    def __new__(cls, config: TabularProjectionConfig):
        if config.input_cardinalities:
            return TabularInputProjection(config)
        elif config.output_cardinalities:
            return TabularOutputProjection(config)
        else:
            raise ValueError("Either input_cardinalities or output_cardinalities must be provided in the config.")