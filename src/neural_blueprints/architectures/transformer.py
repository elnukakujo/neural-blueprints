import torch
import torch.nn as nn

from ..components.composite import TransformerEncoder, TransformerDecoder
from ..components.core import ProjectionLayer
from ..config import TransformerConfig

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