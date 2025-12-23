from typing import List, Optional
from pydantic import BaseModel, model_validator

from ..components.composite.projections.input import TabularInputProjectionConfig
from ..components.composite.projections.output import TabularOutputProjectionConfig

class MLPConfig(BaseModel):
    """Configuration for a Multi-Layer Perceptron (MLP) architecture.
    
    Args:
        input_dim (Optional[int]): Dimension of the input features. Not necessary if using an input projection.
        hidden_dims (List[int]): List of hidden layer dimensions.
        output_dim (Optional[int]): Dimension of the output features. Not necessary if using an output projection.
        normalization (Optional[str]): Normalization to apply after each layer.
        activation (Optional[str]): Activation function to apply after each layer.
        final_activation (Optional[str]): Activation function to apply after the final layer.
        input_projection (Optional[str]): Type of input projection to use.
        output_projection (Optional[str]): Type of output projection to use.
    """

    input_dim: Optional[int] = None
    hidden_dims: List[int]
    output_dim: Optional[int] = None
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None
    final_activation: Optional[str] = None
    input_projection: Optional[TabularInputProjectionConfig] = None
    output_projection: Optional[TabularOutputProjectionConfig] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_projection is None and self.input_dim is None:
            raise ValueError("Either input_projection or input_dim must be specified")
        if self.input_dim is not None and self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_projection is None and self.output_dim is None:
            raise ValueError("Either output_projection or output_dim must be specified")
        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if any(h <= 0 for h in self.hidden_dims):
            raise ValueError("All hidden_dims must be positive integers")
        if self.activation is not None and self.activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu', 'softmax'):
            raise ValueError(f"Unsupported activation: {self.activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu', 'softmax'}")
        if self.dropout_p is not None and not (0 <= self.dropout_p < 1.0):
            raise ValueError(f"Unsupported dropout_p, should be between 0 or 1 but got: {self.dropout_p}")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu', 'softmax'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu', 'softmax'}")
        return self