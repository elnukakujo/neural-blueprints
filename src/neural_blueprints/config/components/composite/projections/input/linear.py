from typing import List
from pydantic import model_validator

from .base import BaseProjectionInputConfig

class LinearInputProjectionConfig(BaseProjectionInputConfig):
    input_dim: List[int]

    @model_validator(mode='after')
    def _validate(self):
        return self