from typing import List, Optional
from pydantic import BaseModel, model_validator

from .base import BaseCompositeConfig
from ..core.base import BaseCoreConfig

class EncoderConfig(BaseCompositeConfig):
    """Configuration for an encoder composed of multiple layers.

    Args:
        layer_configs (List[DenseLayerConfig | ConvLayerConfig | RecurrentUnitConfig | AttentionLayerConfig | ReshapeLayerConfig | NormalizationConfig | FlattenLayerConfig]): List of layer configurations.
        normalization (str | None): Configuration for normalization layer. If None, no normalization is applied.
        activation (str | None): Activation function to use. If None, no activation is applied.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
        final_activation (Optional[str]): Optional final activation function.
    """
    layer_configs: List[BaseCoreConfig]

    @model_validator(mode='after')
    def _validate(self):
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self
    
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