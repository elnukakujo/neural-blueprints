from pydantic import model_validator
from typing import List

from .base import BaseProjectionInputConfig

class TabularInputProjectionConfig(BaseProjectionInputConfig):
    """
    Configuration for tabular data input projection.

    Args:
        latent_dim (int): Dimension of the latent representation.
        hidden_dims (Optional[List[int]]): List of hidden dimensions for the feedforward networks.
        dropout_p (Optional[float]): Dropout probability.
        normalization (Optional[str]): Normalization method to use.
        activation (Optional[str]): Activation function to use.
        cardinalities (List[int]): List of cardinalities for each tabular attribute.
    """ 
    cardinalities: List[int]

    @model_validator(mode='after')
    def _validate(self):
        return self