from typing import List, Optional
from pydantic import model_validator

from .base import BaseProjectionConfig

class TabularProjectionConfig(BaseProjectionConfig):
    input_cardinalities: Optional[List[int]] = None
    output_cardinalities: Optional[List[int]] = None

    input_dim: Optional[List[int]] = None
    output_dim: Optional[List[int]] = None

    projection_dim: Optional[int] = None

    @model_validator(mode='after')
    def _validate(self):
        # Ensure at least one of input_cardinalities or output_cardinalities is provided, and that
        # input_dim and output_dim are provided accordingly.
        
        if self.input_cardinalities is None and self.output_cardinalities is None:
            raise ValueError("At least one of 'input_cardinalities' or 'output_cardinalities' must be provided.")
        
        if self.input_cardinalities is None and self.input_dim is None:
            raise ValueError("'input_dim' or 'input_cardinalities' must be provided.")
        if self.output_cardinalities is None and self.output_dim is None:
            raise ValueError("'output_dim' or 'output_cardinalities' must be provided.")
        
        if self.input_dim is not None and self.output_dim is not None:
            raise ValueError("Only one of 'input_dim' or 'output_dim' should be provided.")
        
        return self