from typing import List, Optional
from pydantic import BaseModel, model_validator

class FeedForwardNetworkConfig(BaseModel):
    """Configuration for a feedforward neural network.
    
    Args:
       input_dim (int): Size of the input features.
       hidden_dims (List[int]): List of hidden layer sizes.
       output_dim (int): Size of the output features.
       normalization (Optional[str]): Normalization type to use in dense layers.
       activation (Optional[str]): Activation function to use in dense layers.
       dropout_p (Optional[float]): Dropout probability to use in dense layers.
       final_activation (Optional[str]): Activation function for the final layer.
    """

    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None
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
        return self