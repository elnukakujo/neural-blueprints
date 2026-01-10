from abc import ABC
from typing import List, Optional
from pydantic import BaseModel, model_validator

class BaseCoreConfig(BaseModel, ABC):
    """Base configuration for neural architectures.

    Args:
        - input_dim (List[int]): Dimensions of the input features.
        - output_dim (Optional[List[int]]): Dimensions of the output features. If None, uses input_dim.
        - dropout_p (Optional[float]): Dropout probability to apply in projections if not already set.
        - normalization (Optional[str]): Normalization type to apply in projections if not already set.
        - activation (Optional[str]): Activation function to apply in projections if not already set.
        - final_activation (Optional[str]): Final activation function to apply to the output.
    """
    input_dim: Optional[List[int]] = None
    output_dim: Optional[List[int]] = None
    dropout_p: Optional[float] = None
    normalization: Optional[str] = None
    activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.output_dim is None:
            self.output_dim = self.input_dim
        return self
