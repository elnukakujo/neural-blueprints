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
    output_dim: int

    @model_validator(mode='after')
    def _validate(self):
        return self