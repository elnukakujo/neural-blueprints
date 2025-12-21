from typing import Optional
from pydantic import BaseModel, model_validator

class DenseLayerConfig(BaseModel):
    """Configuration for a dense (fully connected) layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        normalization (str | None): Configuration for normalization layer. If None, no normalization is applied.
        activation (str | None): Activation function to use. Options: 'relu', 'tanh', 'sigmoid'. If None, no activation is applied.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
    """
    input_dim: int
    output_dim: int
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if self.activation is not None and self.activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported activation: {self.activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self