from typing import Optional
from pydantic import model_validator

from .base import BaseArchitectureConfig
from ..components.composite import TransformerDecoderConfig, TransformerEncoderConfig

class TransformerConfig(BaseArchitectureConfig):
    """Configuration for a Transformer architecture.
    
    Args:
        encoder_config (TransformerEncoderConfig): Configuration for the transformer encoder.
        decoder_config (TransformerDecoderConfig): Configuration for the transformer decoder.
    """
    encoder_config: TransformerEncoderConfig
    decoder_config: TransformerDecoderConfig
    
class BERTConfig(BaseArchitectureConfig):
    encoder_layers: int

    @model_validator(mode='after')
    def _validate(self):
        BaseArchitectureConfig._validate(self)
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self