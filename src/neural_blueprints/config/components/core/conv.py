from typing import Optional
from pydantic import BaseModel, model_validator

from .base import BaseCoreConfig

class ConvLayerConfig(BaseCoreConfig):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: Optional[int] = None
    output_padding: Optional[int] = None
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    dim: int = 1 # 1 for Conv1D, 2 for Conv2D, 3 for Conv3D

    @model_validator(mode='after')
    def _validate(self):
        if self.in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        if self.padding is not None and self.padding < 0:
            raise ValueError("padding must be a non-negative integer")
        if self.output_padding is not None and self.output_padding < 0:
            raise ValueError("output_padding must be a non-negative integer")
        if self.dilation <= 0:
            raise ValueError("dilation must be a positive integer")
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        if self.activation is not None and self.activation.lower() not in ('relu', 'leakyrelu', 'elu', 'silu', 'gelu', 'sigmoid', 'tanh'):
            raise ValueError(f"Unsupported activation: {self.activation}. Supported: {'relu', 'leakyrelu', 'elu', 'silu', 'gelu', 'sigmoid', 'tanh'}")
        if self.dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2, or 3")
        return self