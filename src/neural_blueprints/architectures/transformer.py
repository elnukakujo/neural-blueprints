import torch
import torch.nn as nn
import numpy as np

from ..components.composite import TransformerEncoder, TransformerDecoder
from ..components.core import ProjectionLayer, EmbeddingLayer, DenseLayer
from ..config import TransformerConfig, BERTConfig, DenseLayerConfig, EmbeddingLayerConfig, NormalizationConfig
from ..utils import get_normalization

class Transformer(nn.Module):
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
        """
        src: (batch_size, src_seq_len, input_dim)
        tgt: (batch_size, tgt_seq_len, input_dim)
        """
        # Encode
        memory = self.encoder(src)  # (batch_size, src_seq_len, hidden_dim)

        # Decode (conditioned on memory)
        decoded = self.decoder(tgt, memory)  # (batch_size, tgt_seq_len, hidden_dim)

        # Optional output projection
        out = self.output_proj(decoded)
        return out
    
class BERT(nn.Module):
    """
    A BERT-style model for masked attribute inference on tabular data:
    - Embeddings for categorical features
    - Pass-through for continuous features
    - TransformerEncoder
    - Optional feedforward head for masked attribute prediction
    """
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config

        # Configuration
        self.vocab_size = config.vocab_size            # categorical vocab size
        self.max_seq_len = config.max_seq_len          # number of features
        self.cat_cardinalities = config.cat_cardinalities                    # boolean tensor for categorical features
        self.is_cat = np.array(config.is_cat)
        self.hidden_dim = config.encoder_config.hidden_dim
        self.encoder_config = config.encoder_config
        self.num_cat = len(self.cat_cardinalities)
        self.num_cont = self.max_seq_len - self.num_cat


        # ---- Embeddings for categorical features ----
        self.cat_embeddings = nn.ModuleList([
            EmbeddingLayer(config=EmbeddingLayerConfig(num_embeddings=cardinality+1, embedding_dim=self.hidden_dim, padding_idx=0))
            for cardinality in self.cat_cardinalities
        ])
        self.num_proj = DenseLayer(DenseLayerConfig(input_dim=1, output_dim=self.hidden_dim)) if self.num_cont > 0 else nn.Identity()
        # Positional embeddings (optional)
        self.pos_emb = EmbeddingLayer(
            config=EmbeddingLayerConfig(num_embeddings=self.max_seq_len, embedding_dim=self.hidden_dim)
        )

        self.emb_norm = get_normalization(
            NormalizationConfig(norm_type="layernorm", num_features=self.hidden_dim)
        )
        self.dropout = nn.Dropout(config.dropout)

        # ---- Transformer Encoder ----
        self.encoder = TransformerEncoder(self.encoder_config)

        # ---- Heads for masked attribute prediction ----
        self.cat_heads = nn.ModuleList(
            DenseLayer(config=DenseLayerConfig(input_dim=self.hidden_dim, output_dim=cardinality))
            for cardinality in self.cat_cardinalities
        )
        self.cont_head = DenseLayer(config=DenseLayerConfig(input_dim=self.hidden_dim, output_dim=self.num_cont)) if self.num_cont > 0 else nn.Identity()

    def blueprint(self):
        print(self)
        return self.config

    def forward(self, x, masked_positions):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        # ---- Split categorical and continuous features ----
        cat_x = x[:, self.is_cat].long()
        cont_x = x[:, ~self.is_cat].float()

        # ---- Categorical embeddings ----
        cat_emb_list = []
        for i, emb in enumerate(self.cat_embeddings):
            try:
                cat_emb_list.append(emb(cat_x[:, i]+1))
            except IndexError as e:
                raise IndexError(f"IndexError for categorical feature {i} with cardinality {self.cat_cardinalities[i]} and input value {cat_x[:, i]+1}") from e
        cat_emb = torch.stack(cat_emb_list, dim=1)  # shape: (batch, num_cat, hidden_dim)

        # ---- Concatenate continuous features (projected to hidden_dim) ----
        if self.num_cont > 0:
            cont_x = cont_x.unsqueeze(-1)  # shape: (B, num_cont, 1)
            cont_x = self.num_proj(cont_x)  # shape: (B, num_cont, hidden_dim)
            x_emb = torch.cat([cat_emb, cont_x], dim=1)  # broadcast to (B, num_cat, hidden_dim)
        else:
            x_emb = cat_emb

        # ---- Add positional embeddings ----
        x_emb = x_emb + self.pos_emb(pos)

        x_emb = self.emb_norm(x_emb)
        x_emb = self.dropout(x_emb)

        attn_mask = ~masked_positions          # True = keep, False = ignore
        # Convert to float mask for nn.MultiheadAttention (optional)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf'))

        # ---- Transformer encoder ----
        hidden = self.encoder(x_emb, attn_mask)  # shape: (B, num_features, hidden_dim)

        # ---- Heads ----
        cat_logits = [head(hidden[:, i]) for i, head in enumerate(self.cat_heads)]
        cont_preds = self.cont_head(hidden[:, -self.num_cont:])

        # ---- Masked positions ----
        cat_logits = [logit[masked_positions[:, i]] for i, logit in enumerate(cat_logits)]
        cont_preds = cont_preds[masked_positions[:, ~self.is_cat]]
                
        return cat_logits, cont_preds