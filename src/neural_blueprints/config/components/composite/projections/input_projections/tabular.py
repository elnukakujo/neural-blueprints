from pydantic import BaseModel, model_validator
from typing import Optional, List

class TabularInputProjectionConfig(BaseModel):
    """
    Configuration for tabular input projection.

    Args:
        cardinalities (List[int]): List of cardinalities for each categorical attribute.
        hidden_dims (Optional[List[int]]): List of hidden dimensions for the feedforward networks.
        output_dim (int): Dimension of the output representation.
        dropout_p (Optional[float]): Dropout probability.
        normalization (Optional[str]): Normalization method to use.
        activation (Optional[str]): Activation function to use.
    """    
    cardinalities: List[int]
    hidden_dims: Optional[List[int]]
    output_dim: int
    dropout_p: Optional[float] = None
    normalization: Optional[str] = None
    activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        return self