from pydantic import BaseModel, model_validator
from typing import Optional, List

class TabularOutputProjectionConfig(BaseModel):
    """
    Configuration for Tabular Output Projection.

    Args:
        cardinalities (List[int]): List of cardinalities for each categorical attribute.
        latent_dim (int): Dimension of the latent representation.
        hidden_dims (List[int]): List of hidden dimensions for the feedforward networks.
        activation (Optional[str]): Activation function to use.
        normalization (Optional[str]): Normalization method to use.
        dropout_p (Optional[float]): Dropout probability.
        final_activation (Optional[str]): Final activation function to use.
    """
    cardinalities: List[int]
    input_dim: int
    hidden_dims: List[int]
    activation: Optional[str] = None
    normalization: Optional[str] = None
    dropout_p: Optional[float] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        return self