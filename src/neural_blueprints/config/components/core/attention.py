from pydantic import model_validator

from .base import BaseCoreConfig

class AttentionLayerConfig(BaseCoreConfig):
    """Configuration for an attention layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
    """
    num_heads: int

    @model_validator(mode='after')
    def _validate(self):
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        return self