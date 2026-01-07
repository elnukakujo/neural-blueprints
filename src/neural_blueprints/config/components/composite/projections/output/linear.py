from typing import Optional, List
from pydantic import model_validator

from .base import BaseProjectionOutputConfig

class LinearOutputProjectionConfig(BaseProjectionOutputConfig):
    """
    Configuration for Linear Output Projection.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dims (List[int]): List of hidden layer dimensions.
        output_dim (int): Dimension of the output features.
        activation (Optional[str]): Activation function type. If None, no activation is applied.
        normalization (Optional[str]): Normalization layer type. If None, no normalization is applied.
        dropout_p (Optional[float]): Dropout probability. If None, no dropout is applied.
    """
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: Optional[str] = None
    normalization: Optional[str] = None
    dropout_p: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self):
        return self