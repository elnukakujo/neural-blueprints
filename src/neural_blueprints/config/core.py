from typing import List, Tuple, Optional, Any, Dict
import torch.nn as nn
from pydantic import BaseModel, model_validator, Field

class DenseLayerConfig(BaseModel):
    """Configuration for a dense (fully connected) layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        activation (str | None): Activation function to use. Options: 'relu', 'tanh', 'sigmoid'. If None, no activation is applied.
    """
    input_dim: int
    output_dim: int
    activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if self.activation is not None and self.activation.lower() not in ('relu', 'tanh', 'sigmoid'):
            raise ValueError(f"Unsupported activation: {self.activation}. Supported: {'relu', 'tanh', 'sigmoid'}")
        return self
    
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
        batch_norm (bool): If True, applies batch normalization after the convolution.
        activation (str | None): Activation function to use. Options: 'relu', 'leakyrelu', 'elu', 'silu', 'gelu', 'sigmoid', 'tanh'. If None, no activation is applied.
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: Optional[int] = None
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    batch_norm: bool = False
    activation: Optional[str] = "relu"

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
        if self.dilation <= 0:
            raise ValueError("dilation must be a positive integer")
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        if self.activation is not None and self.activation.lower() not in ('relu', 'leakyrelu', 'elu', 'silu', 'gelu', 'sigmoid', 'tanh'):
            raise ValueError(f"Unsupported activation: {self.activation}. Supported: {'relu', 'leakyrelu', 'elu', 'silu', 'gelu', 'sigmoid', 'tanh'}")
        return self
    
class RecurrentUnitConfig(BaseModel):
    """Configuration for a recurrent neural network unit.
    
    Args:
        rnn_type (str): Type of RNN. Options: 'LSTM', 'GRU', 'RNN'.
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        bidirectional (bool): If True, becomes a bidirectional RNN.
    """
    input_dim: int
    hidden_dim: int
    num_layers: int = 1
    rnn_type: str = 'RNN'
    bidirectional: bool = False

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if self.rnn_type.upper() not in ('LSTM', 'GRU', 'RNN'):
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}. Supported types: 'LSTM', 'GRU', 'RNN'.")
        return self
    
class AttentionLayerConfig(BaseModel):
    """Configuration for an attention layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
    """
    input_dim: int
    num_heads: int

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        return self
    
class ResidualLayerConfig(BaseModel):
    """Configuration for a residual layer.
    
    Args:
        layer_config (BaseModel): Configuration of the layer to be wrapped with residual connection.
    """
    layer_config: nn.Module

    @model_validator(mode='after')
    def _validate(self):
        if not isinstance(self.layer_config, nn.Module):
            raise ValueError("layer_config must be an instance of nn.Module")
        return self
    
class EmbeddingLayerConfig(BaseModel):
    """Configuration for an embedding layer.
    
    Args:
        num_embeddings (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
    """
    num_embeddings: int
    embedding_dim: int

    @model_validator(mode='after')
    def _validate(self):
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be a positive integer")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        return self
    
class PatchEmbeddingLayerConfig(BaseModel):
    """Configuration for a patch embedding layer.
    
    Args:
        embedding_dim (int): Dimension of the embeddings.
        patch_size (int): Size of each patch.
    """
    embedding_dim: int
    patch_size: int
    in_channels: int = 3

    @model_validator(mode='after')
    def _validate(self):
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        return self
    
class PoolingLayerConfig(BaseModel):
    """Configuration for a pooling layer.
    
    Args:
        pool_type (str): Type of pooling. Options: 'max', 'avg'.
        kernel_size (int): Size of the pooling kernel.
        stride (int): Stride of the pooling operation.
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