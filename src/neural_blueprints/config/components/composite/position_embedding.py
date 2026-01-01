from typing import Optional
from pydantic import BaseModel, model_validator

class PositionEmbeddingConfig(BaseModel):
    """Configuration for a feedforward neural network.
    
    Args:
        num_positions (int): The maximum number of positions (sequence length).
        latent_dim (int): The dimensionality of the positional embeddings.
        normalization (Optional[str]): Type of normalization to apply (e.g., 'layer', 'batch'). Default is None.
        activation (Optional[str]): Activation function to use (e.g., 'relu', 'gelu'). Default is None.
        dropout_p (Optional[float]): Dropout probability. Default is None.
    """

    num_positions: int
    latent_dim: int
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.num_positions <= 0:
            raise ValueError("num_positions must be a positive integer.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if self.dropout_p is not None and not (0.0 <= self.dropout_p < 1.0):
            raise ValueError("dropout_p must be in the range [0.0, 1.0).")
        return self