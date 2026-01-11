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
        elif isinstance(self.input_spec, tuple) and len(self.input_spec) == 2:
            self.input_projection = None
        else:
            raise ValueError("Input projection config must be provided for BERT when input_spec is not a 2D tensor shape.")

        input_dim = self.input_projection.output_dim if self.input_projection is not None else self.input_spec
        print(input_dim)

        assert isinstance(input_dim, (tuple, list)) and len(input_dim) == 2 and isinstance(input_dim[0], int) and isinstance(input_dim[1], int), f"Input dimensions must be a tuple of two integers (num_features, latent_dim) but got: {input_dim}."

        # Positional embeddings
        self.position_embedding = PositionEmbedding(
            config=PositionEmbeddingConfig(
                input_dim=[input_dim[0]],
                output_dim=input_dim,
                normalization=normalization,
                activation=activation,
                dropout_p=dropout_p
            )
        )

        # ---- Transformer Encoder ----
        self.encoder = TransformerEncoder(
            config=TransformerEncoderConfig(
                input_dim=input_dim,
                dim_feedforward=input_dim[1] * 4,
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


    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the BERT encoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            Encoded tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # ---- Split categorical and continuous features ----
        x_embed, nan_mask = self.input_projection(inputs)  # shape: (B, num_features, hidden_dim), (B, num_features)

        # ---- Add positional embeddings ----
        x_embed = x_embed + self.position_embedding(inputs)  # shape: (B, num_features, hidden_dim)

        # ---- Transformer encoder ----
        x_embed = self.encoder(x_embed, attn_mask = nan_mask)  # shape: (B, num_features, hidden_dim)

        return x_embed

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """Forward pass through the BERT model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            List of output tensors for each feature or output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # ---- Split categorical and continuous features ----
        if self.input_projection is not None:
                x_embed, nan_mask = self.input_projection(inputs)  # shape: (B, num_features, hidden_dim), (B, num_features)
        else:
            x_embed = inputs  # shape: (B, num_features, hidden_dim)
            nan_mask = None

        # ---- Add positional embeddings ----
        x_embed = x_embed + self.position_embedding(x_embed)  # shape: (B, num_features, hidden_dim)

        # ---- Transformer encoder ----
        x_embed = self.encoder(x_embed, attn_mask = nan_mask)  # shape: (B, num_features, hidden_dim)

        if self.output_projection is not None:
            # ---- Output projections for each feature ----
            return self.output_projection(x_embed)
        else:
            return x_embed