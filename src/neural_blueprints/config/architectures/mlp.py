from typing import List, Optional
from pydantic import BaseModel, model_validator

class MLPConfig(BaseModel):
    """Configuration for a Multi-Layer Perceptron (MLP) architecture.
    
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dims (List[int]): List of hidden layer dimensions.
        output_dim (int): Dimension of the output features.
        normalization (Optional[str]): Normalization to apply after each layer.
        activation (Optional[str]): Activation function to apply after each layer.
        final_activation (Optional[str]): Activation function to apply after the final layer.
    """

    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    normalization: Optional[str] = None
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