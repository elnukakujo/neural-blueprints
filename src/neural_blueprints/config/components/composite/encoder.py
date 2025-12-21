from typing import List, Optional
from pydantic import BaseModel, model_validator

from ...components.core import (
    DenseLayerConfig,
    ConvLayerConfig,
    RecurrentUnitConfig,
    AttentionLayerConfig,
    ReshapeLayerConfig,
    NormalizationLayerConfig,
    FlattenLayerConfig
)

class EncoderConfig(BaseModel):
    """Configuration for an encoder composed of multiple layers.

    Args:
        layer_configs (List[DenseLayerConfig | ConvLayerConfig | RecurrentUnitConfig | AttentionLayerConfig | ReshapeLayerConfig | NormalizationConfig | FlattenLayerConfig]): List of layer configurations.
        normalization (str | None): Configuration for normalization layer. If None, no normalization is applied.
        activation (str | None): Activation function to use. If None, no activation is applied.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
        final_activation (Optional[str]): Optional final activation function.
    """

    layer_configs: List[DenseLayerConfig | ConvLayerConfig | RecurrentUnitConfig | AttentionLayerConfig | ReshapeLayerConfig | NormalizationLayerConfig | FlattenLayerConfig]
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self
    
class TransformerEncoderConfig(BaseModel):
    """Configuration for a Transformer encoder.

    Args:
        input_dim (int): Size of the input features.
        hidden_dim (int): Size of the hidden layer.
        num_layers (int): Number of layers.
        num_heads (int): Number of attention heads.
        normalization (str | None): Configuration for normalization layer. If None, no normalization is applied.
        activation (str | None): Activation function to use. If None, no activation is applied.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
        final_activation (Optional[str]): Optional final activation function.
    """

    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    activation: Optional[str] = None
    dropout_p: Optional[float] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self