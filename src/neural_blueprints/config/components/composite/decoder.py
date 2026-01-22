from typing import List, Optional
from pydantic import model_validator

from .base import BaseCompositeConfig

class DecoderConfig(BaseCompositeConfig):
    layers_dim: List[int]
    latent_dim: int
    
class TransformerDecoderConfig(BaseCompositeConfig):
    """Configuration for a Transformer decoder.
    
    Args:
        input_dim (int): Size of the input features.
        hidden_dim (int): Size of the hidden layer.
        num_layers (int): Number of layers.
        num_heads (int): Number of attention heads.
        normalization (str | None): Configuration for normalization layer. If None, no normalization is applied.
        activation (str | None): Activation function to use. If None, no activation is applied
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
        final_activation (Optional[str]): Optional final activation function.
    """
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