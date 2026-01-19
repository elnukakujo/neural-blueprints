from typing import List

from .base import BaseProjectionConfig

class LinearProjectionConfig(BaseProjectionConfig):
    """
    Configuration for Linear Output Projection.

    Args:
        input_dim (List[int]): Input tensor dimensions.
        output_dim (List[int]): Output tensor dimensions.
        hidden_dims (List[int]): List of hidden layer dimensions.
        activation (Optional[str]): Activation function type. If None, no activation is applied.
        normalization (Optional[str]): Normalization layer type. If None, no normalization is applied.
        dropout_p (Optional[float]): Dropout probability. If None, no dropout is applied.
    """
    input_dim: List[int]
    output_dim: List[int]