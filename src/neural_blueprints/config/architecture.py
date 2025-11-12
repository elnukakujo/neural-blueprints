from typing import List, Tuple, Optional, Any, Dict
from pydantic import BaseModel, model_validator, Field

class MLPConfig(BaseModel):
    """Configuration for a Multi-Layer Perceptron (MLP) architecture."""

    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: Optional[str] = None
    final_activation: Optional[str] = None

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
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu'}")
        return self
    
class CNNConfig(BaseModel):
    """Configuration for a Convolutional Neural Network (CNN) architecture."""

    final_activation: Optional[str] = None
    layer_types: List[str]
    layer_configs: List[BaseModel]
    feedforward_config: BaseModel

    @model_validator(mode='after')
    def _validate(self):
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu'}")
        for layer_type in self.layer_types:
            if layer_type.lower() not in ('conv1d', 'conv2d', 'pool1d', 'pool2d'):
                raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {'conv1d', 'conv2d', 'pool1d', 'pool2d'}")
        return self
    
class RNNConfig(BaseModel):
    """Configuration for a Recurrent Neural Network (RNN) architecture."""

    rnn_unit_config: BaseModel
    output_dim: int
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu'}")
        return self