from pydantic import model_validator
from typing import List

from .base import BaseProjectionOutputConfig

class TabularOutputProjectionConfig(BaseProjectionOutputConfig):
    """
    Configuration for Tabular Output Projection.

    Args:
        input_dim (List[int]): List of input dimensions.
        hidden_dims (Optional[List[int]]): List of hidden layer dimensions.
        cardinalities (List[int]): List of cardinalities for each tabular attribute.
        activation (Optional[str]): Activation function type. If None, no activation is applied.
        normalization (Optional[str]): Normalization layer type. If None, no normalization is applied.
        dropout_p (Optional[float]): Dropout probability. If None, no dropout is applied.
    """
    cardinalities: List[int]

    @model_validator(mode='after')
    def _validate(self):
        return self