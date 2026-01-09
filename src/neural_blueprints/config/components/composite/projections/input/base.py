from pydantic import BaseModel
from typing import Optional, List

class BaseProjectionInputConfig(BaseModel):
    """
    Base configuration for input projections.

    This class serves as a base for specific input projection configurations.
    It can be extended to include common attributes or methods shared across
    different input projection types.
    Args:
        latent_dim (int): Dimension of the latent representation.
        hidden_dims (Optional[List[int]]): List of hidden dimensions for the feedforward networks.
        dropout_p (Optional[float]): Dropout probability.
        normalization (Optional[str]): Normalization method to use.
        activation (Optional[str]): Activation function to use.
    """
    latent_dim: int
    hidden_dims: Optional[List[int]] = None
    dropout_p: Optional[float] = None
    normalization: Optional[str] = None
    activation: Optional[str] = None