from pydantic import BaseModel, model_validator
from typing import Optional, List

class TabularOutputProjectionConfig(BaseModel):
    """
    Configuration for Tabular Output Projection.

    Args:
        input_cardinalities (List[int]): List of cardinalities for each input categorical attribute.
        output_cardinalities (Optional[List[int]]): List of cardinalities for each output categorical attribute. If none, defaults to input cardinalities.
        input_dim (List[int]): List of input dimensions.
        hidden_dims (List[int]): List of hidden dimensions for the feedforward networks.
        activation (Optional[str]): Activation function to use.
        normalization (Optional[str]): Normalization method to use.
        dropout_p (Optional[float]): Dropout probability.
    """
    input_cardinalities: List[int]
    output_cardinalities: Optional[List[int]] = None
    input_dim: List[int]
    hidden_dims: List[int]
    activation: Optional[str] = None
    normalization: Optional[str] = None
    dropout_p: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self):
        return self