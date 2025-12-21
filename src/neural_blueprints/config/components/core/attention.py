from typing import Optional
from pydantic import BaseModel, model_validator

class AttentionLayerConfig(BaseModel):
    """Configuration for an attention layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
    """
    input_dim: int
    num_heads: int
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        return self