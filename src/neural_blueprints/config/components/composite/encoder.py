from typing import List, Optional
from pydantic import model_validator

from .base import BaseCompositeConfig

class EncoderConfig(BaseCompositeConfig):
    layers_dim: List[int]
    latent_dim: int
    
class TransformerEncoderConfig(BaseCompositeConfig):
    dim_feedforward: int
    num_layers: int
    num_heads: int

    @model_validator(mode='after')
    def _validate(self):
        if self.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self