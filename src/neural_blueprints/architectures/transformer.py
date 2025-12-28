import torch
import torch.nn as nn
from typing import List

from ..components.composite.projections.input import TabularInputProjection
from ..components.composite.projections.output import TabularOutputProjection
from ..components.composite import TransformerEncoder, TransformerDecoder
from ..components.core import EmbeddingLayer, NormalizationLayer, DropoutLayer

from ..config.architectures import TransformerConfig, TabularBERTConfig
from ..config.components.composite import TransformerEncoderConfig
from ..config.components.composite.projections.input import TabularInputProjectionConfig
from ..config.components.composite.projections.output import TabularOutputProjectionConfig
from ..config.components.core import EmbeddingLayerConfig, NormalizationLayerConfig, DropoutLayerConfig

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

    def blueprint(self):
        print(self)
        return self.config

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
    
class TabularBERT(nn.Module):
    """A BERT-style model for masked attribute inference on tabular data:
        - Embeddings for categorical features
        - Pass-through for continuous features
        - TransformerEncoder
        - Optional feedforward head for masked attribute prediction

    Args:
        config (TabularBERTConfig): Configuration for the Tabular BERT model.
    """
    def __init__(self,
            config: TabularBERTConfig    
        ):
        super().__init__()
        self.config = config

        input_cardinalities = config.input_cardinalities
        output_cardinalities = config.output_cardinalities
        latent_dim = config.latent_dim
        encoder_layers = config.encoder_layers
        dropout_p = config.dropout_p
        normalization = config.normalization
        activation = config.activation
        final_activation = config.final_activation

        # ---- Projections for discrete and continuous features ----
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
        self.pos_emb = EmbeddingLayer(
            config=EmbeddingLayerConfig(num_embeddings=len(input_cardinalities)+1, embedding_dim=latent_dim)
        )

        self.emb_norm = NormalizationLayer(
            NormalizationLayerConfig(norm_type=normalization, num_features=latent_dim)
        )
        self.dropout = DropoutLayer(
            config=DropoutLayerConfig(
                p=dropout_p
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
        self.output_projection = TabularOutputProjection(
            config=TabularOutputProjectionConfig(
                input_cardinalities=input_cardinalities,
                output_cardinalities=output_cardinalities,
                input_dim=[len(input_cardinalities), latent_dim],
                hidden_dims=[latent_dim*2, latent_dim*4, latent_dim*8],
                dropout_p=dropout_p,
                normalization=normalization,
                activation=activation,
                final_activation=final_activation
            )
        )

    def blueprint(self):
        print(self)
        return self.config

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """Forward pass through the Tabular BERT model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            List of output tensors for each feature or output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        B, L = x.shape
        # ---- Split categorical and continuous features ----
        x_embed, nan_mask = self.input_projection(x)  # shape: (B, num_features, hidden_dim), (B, num_features)

        # ---- Add positional embeddings ----
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L) # shape: (batch, seq)
        x_embed = x_embed + self.pos_emb(pos)

        x_embed = self.emb_norm(x_embed)
        x_embed = self.dropout(x_embed)

        # ---- Transformer encoder ----
        x_embed = self.encoder(x_embed, attn_mask = nan_mask)  # shape: (B, num_features, hidden_dim)
        assert x_embed.cpu().isnan().sum() == 0, "NaN values detected in Transformer encoder output"

        # ---- Output projections for each feature ----
        return self.output_projection(x_embed)