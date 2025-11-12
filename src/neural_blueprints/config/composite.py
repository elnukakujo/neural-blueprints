from typing import List, Tuple, Optional, Any, Dict
from pydantic import BaseModel, model_validator, Field

class FeedForwardNetworkConfig(BaseModel):
    """Configuration for a feedforward neural network."""

    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if any(h <= 0 for h in self.hidden_dims):
            raise ValueError("All hidden_dims must be positive integers")
        if self.activation is not None and self.activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu'):
            raise ValueError(f"Unsupported activation: {self.activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu'}")
        return self
    
class EncoderConfig(BaseModel):
    """Configuration for an encoder composed of multiple layers."""

    layer_types: List[str]
    layer_configs: List[BaseModel]
    projection_dim: Optional[int] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if len(self.layer_types) != len(self.layer_configs):
            raise ValueError("layer_types and layer_configs must have the same length")
        supported_layers = {'dense', 'conv1d', 'conv2d', 'recurrent', 'attention'}
        for layer_type in self.layer_types:
            if layer_type.lower() not in supported_layers:
                raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {supported_layers}")
        if self.projection_dim is not None and self.projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer if specified")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self
    
class DecoderConfig(BaseModel):
    """Configuration for a decoder composed of multiple layers."""

    layer_types: List[str]
    layer_configs: List[BaseModel]
    projection_dim: Optional[int] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if len(self.layer_types) != len(self.layer_configs):
            raise ValueError("layer_types and layer_configs must have the same length")
        supported_layers = {'dense', 'conv1d_transpose', 'conv2d_transpose', 'attention'}
        for layer_type in self.layer_types:
            if layer_type.lower() not in supported_layers:
                raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {supported_layers}")
        if self.projection_dim is not None and self.projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer if specified")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self
    
class GeneratorConfig(DecoderConfig):
    """Configuration for a generator model."""
    pass
    
class DiscriminatorConfig(EncoderConfig):
    """Configuration for a discriminator model."""
    pass

class TransformerEncoderConfig(BaseModel):
    """Configuration for a Transformer encoder."""

    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float = 0.1
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
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self
    
class TransformerDecoderConfig(BaseModel):
    """Configuration for a Transformer decoder."""

    input_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float = 0.1
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
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self