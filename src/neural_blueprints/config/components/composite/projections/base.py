from pydantic import BaseModel
from typing import Optional, List
from abc import ABC

class BaseProjectionConfig(BaseModel, ABC):
    """
    Base configuration for projection components.
    
    Args:
        - hidden_dims (Optional[List[int]]): List of hidden layer dimensions.
        - dropout_p (Optional[float]): Dropout probability.
        - normalization (Optional[str]): Type of normalization to use.
        - activation (Optional[str]): Activation function to use.
        - final_activation (Optional[str]): Final activation function to use.
    """
    hidden_dims: Optional[List[int]] = None
    dropout_p: Optional[float] = None
    normalization: Optional[str] = None
    activation: Optional[str] = None
    final_activation: Optional[str] = None