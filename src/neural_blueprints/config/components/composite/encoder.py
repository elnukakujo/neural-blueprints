from typing import List, Optional
from pydantic import BaseModel, model_validator

from ...core import ProjectionLayerConfig, NormalizationConfig

class EncoderConfig(BaseModel):
    """Configuration for an encoder composed of multiple layers.

    Args:
        layer_types (List[str]): List of layer types.
        layer_configs (List[BaseModel]): List of layer configurations.
        projection (Optional[ProjectionLayerConfig]): Optional projection layer configuration.
        final_activation (Optional[str]): Optional final activation function.
    """

    layer_types: List[str]
    layer_configs: List[BaseModel]
    projection: Optional[ProjectionLayerConfig] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if len(self.layer_types) != len(self.layer_configs):
            raise ValueError("layer_types and layer_configs must have the same length")
        supported_layers = {'dense', 'conv1d', 'conv2d', 'recurrent', 'attention', 'flatten', 'reshape'}
        for layer_type in self.layer_types:
            if layer_type.lower() not in supported_layers:
                raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {supported_layers}")
        if self.projection is not None:
            if self.projection.input_dim <= 0 or self.projection.output_dim <= 0:
                raise ValueError("projection input_dim and output_dim must be positive integers if specified")
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
    """

    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float = 0.1
    projection: Optional[ProjectionLayerConfig] = None
    final_normalization: Optional[NormalizationConfig] = None
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
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in the range [0.0, 1.0)")
        if self.projection is not None:
            if self.projection.input_dim <= 0 or self.projection.output_dim <= 0:
                raise ValueError("projection input_dim and output_dim must be positive integers if specified")
        if self.final_normalization is not None and self.final_normalization.norm_type.lower() not in ('batchnorm1d', 'batchnorm2d', 'layernorm'):
            raise ValueError(f"Unsupported final_normalization: {self.final_normalization.norm_type}. Supported: {'batchnorm1d', 'batchnorm2d', 'layernorm'}")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self