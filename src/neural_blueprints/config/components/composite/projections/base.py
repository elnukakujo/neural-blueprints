from pydantic import BaseModel
from typing import Optional, List
from abc import ABC

class BaseProjectionConfig(BaseModel, ABC):
    hidden_dims: Optional[List[int]] = None
    dropout_p: Optional[float] = None
    normalization: Optional[str] = None
    activation: Optional[str] = None