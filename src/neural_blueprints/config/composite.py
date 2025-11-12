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

    @model_validator(mode='after')
    def _validate(self):
        if len(self.layer_types) != len(self.layer_configs):
            raise ValueError("layer_types and layer_configs must have the same length")
        supported_layers = {'dense', 'conv1d', 'conv2d', 'recurrent', 'attention'}
        for layer_type in self.layer_types:
            if layer_type.lower() not in supported_layers:
                raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {supported_layers}")
        return self
    
class DecoderConfig(BaseModel):
    """Configuration for a decoder composed of multiple layers."""

    layer_types: List[str]
    layer_configs: List[BaseModel]

    @model_validator(mode='after')
    def _validate(self):
        if len(self.layer_types) != len(self.layer_configs):
            raise ValueError("layer_types and layer_configs must have the same length")
        supported_layers = {'dense', 'conv1d_transpose', 'conv2d_transpose', 'attention'}
        for layer_type in self.layer_types:
            if layer_type.lower() not in supported_layers:
                raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {supported_layers}")
        return self
    
class GeneratorConfig(BaseModel):
    """Configuration for a generator model."""
    
    latent_dim: int
    output_shape: Tuple[int, ...]
    architecture: DecoderConfig

    @model_validator(mode='after')
    def _validate(self):
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        if not all(dim > 0 for dim in self.output_shape):
            raise ValueError("All dimensions in output_shape must be positive integers")
        return self