import torch
import torch.nn as nn
from typing import List

from ..components.composite import TransformerEncoder, TransformerDecoder, FeedForwardNetwork
from ..components.core import ProjectionLayer, EmbeddingLayer
from ..config import TransformerConfig, TabularBERTConfig, EmbeddingLayerConfig, NormalizationConfig, FeedForwardNetworkConfig
from ..utils import get_normalization

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
        
        # Optional final projection (e.g. to regression targets or vocab size)
        self.output_dim = config.output_dim or config.decoder_config.hidden_dim
        self.output_proj = ProjectionLayer(
            input_dim=config.decoder_config.hidden_dim,
            output_dim=self.output_dim
        ) if self.output_dim != config.decoder_config.hidden_dim else nn.Identity()

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

        # Optional output projection
        out = self.output_proj(decoded)
        return out
    
class TabularBERT(nn.Module):
    """A BERT-style model for masked attribute inference on tabular data:
        - Embeddings for categorical features
        - Pass-through for continuous features
        - TransformerEncoder
        - Optional feedforward head for masked attribute prediction

    Args:
        config (TabularBERTConfig): Configuration for the Tabular BERT model.
    """
    def __init__(self, config: TabularBERTConfig):
        super().__init__()
        self.config = config

        # Configuration
        self.seq_len = config.encoder_config.input_dim          # number of features
        self.cardinalities = config.cardinalities                    # boolean tensor for categorical features
        self.hidden_dim = config.encoder_config.hidden_dim
        self.encoder_config = config.encoder_config
        self.with_input_projection = config.with_input_projection
        self.with_output_projection = config.with_output_projection
        self.num_dis = sum(1 for cardinality in self.cardinalities if cardinality > 1)
        self.num_cont = self.seq_len - self.num_dis

        # ---- Projections for discrete and continuous features ----
        if self.with_input_projection:
            self.input_projections = nn.ModuleList([])
            for cardinality in self.cardinalities:
                if cardinality>1:   # Discrete Scenario
                    self.input_projections.append(EmbeddingLayer(config=EmbeddingLayerConfig(num_embeddings=cardinality, embedding_dim=self.hidden_dim)))
                else:               # Continuous Scenario
                    self.input_projections.append(FeedForwardNetwork(config=FeedForwardNetworkConfig(input_dim=1, hidden_dims=[self.hidden_dim, self.hidden_dim], output_dim=self.hidden_dim, activation='relu', dropout=0.1)))

        # Positional embeddings (optional)
        self.pos_emb = EmbeddingLayer(
            config=EmbeddingLayerConfig(num_embeddings=self.seq_len, embedding_dim=self.hidden_dim)
        )

        self.emb_norm = get_normalization(
            NormalizationConfig(norm_type="layernorm", num_features=self.hidden_dim)
        )
        self.dropout = nn.Dropout(config.dropout)

        # ---- Transformer Encoder ----
        self.encoder = TransformerEncoder(self.encoder_config)

        # ---- Heads for masked attribute prediction ----
        if self.with_output_projection:
            self.output_projections = nn.ModuleList([])
            for cardinality in self.cardinalities:
                if cardinality > 1:  # Categorical
                    self.output_projections.append(
                        FeedForwardNetwork(config=FeedForwardNetworkConfig(
                            input_dim=self.hidden_dim,
                            hidden_dims=[self.hidden_dim // 2],
                            output_dim=cardinality
                        ))
                    )
                else:  # Continuous - deeper head
                    self.output_projections.append(
                        FeedForwardNetwork(config=FeedForwardNetworkConfig(
                            input_dim=self.hidden_dim,
                            hidden_dims=[self.hidden_dim, self.hidden_dim // 2],  # Deeper!
                            output_dim=1,
                            activation='relu',
                            dropout=0.1
                        ))
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
        
        if self.with_input_projection:
            x_embed = []
            nan_mask = []
            for column_idx in range(self.seq_len):
                column_data = x[:, column_idx]  # shape: (batch,)
                if self.cardinalities[column_idx] > 1:  # Discrete feature
                    x_embed.append(self.input_projections[column_idx](column_data))  # shape: (batch, hidden_dim)
                    nan_mask.append(column_data == 0)
                else:
                    nan_mask.append(column_data == -2.0)
                    column_data = column_data.unsqueeze(-1)  # shape: (batch, 1)
                    x_embed.append(self.input_projections[column_idx](column_data))  # shape: (batch, hidden_dim)

            nan_mask = torch.stack(nan_mask, dim=1)                 # shape: (batch, seq_len)
            nan_mask = nan_mask.masked_fill(nan_mask == True, float('-inf'))  # shape: (batch, seq_len)

            x_embed = torch.stack(x_embed, dim=1)                   # shape: (batch, seq_len, hidden_dim)
            assert x_embed.shape == (B, L, self.hidden_dim)
        else:
            x_embed = x.repeat(1,1,self.hidden_dim)  # shape: (batch, seq_len, hidden_dim)

        # ---- Add positional embeddings ----
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L) # shape: (batch, seq)
        x_embed = x_embed + self.pos_emb(pos)

        x_embed = self.emb_norm(x_embed)
        x_embed = self.dropout(x_embed)

        # ---- Transformer encoder ----
        x_embed = self.encoder(x_embed, attn_mask = nan_mask)  # shape: (B, num_features, hidden_dim)
        assert x_embed.shape == (B, L, self.hidden_dim)
        assert x_embed.cpu().isnan().sum() == 0, "NaN values detected in Transformer encoder output"

        # ---- Output projections for each feature ----
        if self.with_output_projection:
            predictions = []
            for column_idx in range(self.seq_len):
                predictions.append(self.output_projections[column_idx](x_embed[:, column_idx])) # shape: (batch, output_dim)
            return predictions
        else:
            return x_embed