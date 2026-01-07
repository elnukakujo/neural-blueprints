import torch
import torch.nn as nn
from typing import List

from .base import EncoderArchitecture
from ..components.composite.projections.input import TabularInputProjection
from ..components.composite.projections.output import TabularOutputProjection
from ..components.composite import TransformerEncoder, TransformerDecoder, PositionEmbedding

from ..config.architectures import TransformerConfig, BERTConfig
from ..config.components.composite import TransformerEncoderConfig, PositionEmbeddingConfig
from ..config.components.composite.projections.input import TabularInputProjectionConfig
from ..config.components.composite.projections.output import TabularOutputProjectionConfig

import logging
logger = logging.getLogger(__name__)

class Transformer(nn.Module):
    """A simple Transformer architecture with encoder and decoder.
    
    Args:
        config (TransformerConfig): Configuration for the Transformer model.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config.encoder_config)
        self.decoder = TransformerDecoder(config.decoder_config)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len, input_dim).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_seq_len, input_dim).

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, output_dim).
        """
        # Encode
        memory = self.encoder(src)  # (batch_size, src_seq_len, hidden_dim)

        # Decode (conditioned on memory)
        decoded = self.decoder(tgt, memory)  # (batch_size, tgt_seq_len, hidden_dim)
        return decoded
    
class BERT(EncoderArchitecture):
    """A BERT-style model for masked attribute inference on tabular data:
        - Embeddings for categorical features
        - Pass-through for continuous features
        - TransformerEncoder
        - Optional feedforward head for masked attribute prediction

    Args:
        config (BERTConfig): Configuration for the Tabular BERT model.
    """
    def __init__(self,
            config: BERTConfig    
        ):
        from ..utils import get_input_projection, get_output_projection
        super().__init__()
        self.config = config

        input_cardinalities = config.input_cardinalities
        output_cardinalities = config.output_cardinalities

        self.input_dim = [len(input_cardinalities)]
        self.output_dim = [len(output_cardinalities)]

        latent_dim = config.latent_dim
        encoder_layers = config.encoder_layers
        dropout_p = config.dropout_p
        normalization = config.normalization
        activation = config.activation
        final_activation = config.final_activation

        # ---- Projections for discrete and continuous features ----
        if self.config.input_projection is not None:
            self.input_projection = get_input_projection(
                projection_config=config.input_projection,
            )
        else:
            self.input_projection = None

        self.input_projection = TabularInputProjection(
            config=TabularInputProjectionConfig(
                cardinalities=input_cardinalities,
                hidden_dims=[latent_dim*8, latent_dim*4, latent_dim*2],
                output_dim=[len(input_cardinalities), latent_dim],
                dropout_p=dropout_p,
                normalization=normalization,
                activation=activation
            )
        )

        # Positional embeddings
        self.position_embedding = PositionEmbedding(
            config=PositionEmbeddingConfig(
                num_positions=len(input_cardinalities),
                latent_dim=latent_dim,
                normalization=normalization,
                activation=activation,
                dropout_p=dropout_p
            )
        )

        # ---- Transformer Encoder ----
        self.encoder = TransformerEncoder(
            config=TransformerEncoderConfig(
                input_dim=latent_dim,
                hidden_dim=latent_dim,
                num_layers=encoder_layers,
                num_heads=8,
                dropout_p=dropout_p,
                activation=activation
            )
        )

        # ---- Heads for masked attribute prediction ----
        if config.output_projection is not None:
            self.output_projection = get_output_projection(
                projection_config=config.output_projection,
            )
        else:
            self.output_projection = None


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the BERT encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            Encoded tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # ---- Split categorical and continuous features ----
        x_embed, nan_mask = self.input_projection(x)  # shape: (B, num_features, hidden_dim), (B, num_features)

        # ---- Add positional embeddings ----
        x_embed = x_embed + self.position_embedding(x)  # shape: (B, num_features, hidden_dim)

        # ---- Transformer encoder ----
        x_embed = self.encoder(x_embed, attn_mask = nan_mask)  # shape: (B, num_features, hidden_dim)

        return x_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """Forward pass through the BERT model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            List of output tensors for each feature or output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # ---- Split categorical and continuous features ----
        if self.input_projection is not None:
                x_embed, nan_mask = self.input_projection(x)  # shape: (B, num_features, hidden_dim), (B, num_features)
        else:
            x_embed = x
            nan_mask = None
            
            # If no input projection, ensure x_embed is 3D for positional embedding and encoder
            if x_embed.dim() == 2:
                # Add feature dimension (assume input is (B, F) -> (B, F, 1))
                x_embed = x_embed.unsqueeze(-1)
            elif x_embed.dim() != 3:
                raise ValueError(f"Input tensor must be 2D or 3D, got shape {x_embed.shape}")

        # ---- Add positional embeddings ----
        x_embed = x_embed + self.position_embedding(x)  # shape: (B, num_features, hidden_dim)

        # ---- Transformer encoder ----
        x_embed = self.encoder(x_embed, attn_mask = nan_mask)  # shape: (B, num_features, hidden_dim)

        # ---- Output projections for each feature ----
        return self.output_projection(x_embed)