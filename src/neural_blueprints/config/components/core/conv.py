from typing import Optional
from pydantic import BaseModel, model_validator

class ConvLayerConfig(BaseModel):
    """Configuration for a convolutional layer.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int | None): Padding added to both sides of the input. If None, uses "same" padding.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        activation (Optional[str]): Activation function to use. Options: 'relu', 'leakyrelu', 'elu', 'silu', 'gelu', 'sigmoid', 'tanh'. If None, no activation is applied.
        dropout_p (Optional[float]): Dropout probability. If None, no dropout is applied.
        dim (int): Dimensionality of the convolution (1D, 2D, or 3D).
    """
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
    activation: Optional[str] = None
    dropout_p: Optional[float] = None

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