from typing import Optional
from pydantic import BaseModel, model_validator

from .base import BaseCoreConfig

class PoolingLayerConfig(BaseCoreConfig):
    """Configuration for a pooling layer.
    
    Args:
        pool_type (str): Type of pooling. Options: 'max', 'avg'.
        kernel_size (int): Size of the pooling kernel.
        stride (int): Stride of the pooling operation.
        normalization (str | None): Configuration for normalization layer. If None, no normalization is applied.
        activation (str | None): Activation function to use. If None, no activation is applied.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
    """
    pool_type: str
    kernel_size: int
    stride: int

    @model_validator(mode='after')
    def _validate(self):
        if self.pool_type.lower() not in ('max', 'avg'):
            raise ValueError(f"Unsupported pool_type: {self.pool_type}. Supported types: 'max', 'avg'")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        return self