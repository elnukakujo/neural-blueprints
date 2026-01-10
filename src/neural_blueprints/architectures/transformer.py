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

        self.input_spec = config.input_spec
        self.output_spec = config.output_spec
        
        encoder_layers = config.encoder_layers
        dropout_p = config.dropout_p
        normalization = config.normalization
        activation = config.activation

        # ---- Projections for discrete and continuous features ----
        if config.input_projection is not None:            
            self.input_projection = get_input_projection(
                projection_config=config.input_projection,
            )
            latent_dim = config.input_projection.latent_dim
        else:
            self.input_projection = None
            latent_dim = self.input_dim[-1]

        # Positional embeddings
        self.position_embedding = PositionEmbedding(
            config=PositionEmbeddingConfig(
                input_dim=self.input_dim,
                output_dim=[self.input_dim[0], latent_dim],
                normalization=normalization,
                activation=activation,
                dropout_p=dropout_p
            )
        )

        # ---- Transformer Encoder ----
        self.encoder = TransformerEncoder(
            config=TransformerEncoderConfig(
                input_dim=[self.input_dim[0], latent_dim],
                dim_feedforward=latent_dim * 4,
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

        # ---- Add positional embeddings ----
        x_embed = x_embed + self.position_embedding(x)  # shape: (B, num_features, hidden_dim)

        # ---- Transformer encoder ----
        x_embed = self.encoder(x_embed, attn_mask = nan_mask)  # shape: (B, num_features, hidden_dim)

        if self.output_projection is not None:
            # ---- Output projections for each feature ----
            return self.output_projection(x_embed)
        else:
            return x_embed