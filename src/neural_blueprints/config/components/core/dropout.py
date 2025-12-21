from typing import Optional
from pydantic import BaseModel, model_validator

class DropoutLayerConfig(BaseModel):
    """
    Configuration for a Dropout layer.

    Args:
        p (float): Dropout probability.
    """
    p: Optional[float] = 0.0

    @model_validator(mode="after")
    def check_probability(self):
        if self.p is None:
            self.p = 0.0
        elif not (0.0 <= self.p <= 1.0):
            raise ValueError(f"Dropout probability 'p' must be between 0 and 1 but got:{self.p}")
        return self