from pydantic import model_validator
from typing import List, Optional
from .base import BaseArchitectureConfig

class AutoEncoderConfig(BaseArchitectureConfig):
    encoder_layers: Optional[List[int]] = None
    decoder_layers: Optional[List[int]] = None
    latent_dim: int
    symmetric: Optional[bool] = True
    encoder_attention_layers: Optional[List[int]] = None
    decoder_attention_layers: Optional[List[int]] = None
    latent_attention: Optional[int] = None

    @model_validator(mode='after')
    def _validate_autoencoder(self):
        if not(self.encoder_layers or self.decoder_layers):
            raise ValueError("At least one of encoder_layers or decoder_layers must be specified.")
        if not self.symmetric and not(self.encoder_layers and self.decoder_layers):
            raise ValueError("When symmetric is False, both encoder_layers and decoder_layers must be specified.")
        if self.symmetric and (self.encoder_layers and self.decoder_layers):
            raise ValueError("When symmetric is True, only one of encoder_layers or decoder_layers should be specified.")
        return self