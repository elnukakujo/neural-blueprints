from pydantic import BaseModel
from typing import Optional, List
from abc import ABC

class BaseProjectionOutputConfig(BaseModel, ABC):
    """
    Base configuration for output projections.

    This class serves as a base for specific output projection configurations.
    It can be extended to include common attributes or methods shared across
    different output projection types.

    Args:
        input_dim (List[int]): List of input dimensions.
        hidden_dims (Optional[List[int]]): List of hidden layer dimensions.
        activation (Optional[str]): Activation function type. If None, no activation is applied.
        normalization (Optional[str]): Normalization layer type. If None, no normalization is applied.
        dropout_p (Optional[float]): Dropout probability. If None, no dropout is applied.
    """
    input_dim: List[int]
    hidden_dims: Optional[List[int]] = None
    activation: Optional[str] = None
    normalization: Optional[str] = None
    dropout_p: Optional[float] = None