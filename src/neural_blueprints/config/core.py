from typing import Optional
from pydantic import BaseModel, model_validator

class NormalizationConfig(BaseModel):
    """Configuration for a normalization layer.
    
    Args:
        norm_type (str): Type of normalization. Options: 'batchnorm1d', 'batchnorm2d', 'layernorm'.
        num_features (int): Number of features/channels for the normalization layer.
    """
    norm_type: str | None
    num_features: int

    @model_validator(mode='after')
    def _validate(self):
        if self.norm_type is not None and self.norm_type.lower() not in ('batchnorm1d', 'batchnorm2d', 'layernorm'):
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Supported types: 'batchnorm1d', 'batchnorm2d', 'layernorm'")
        if self.num_features <= 0:
            raise ValueError("num_features must be a positive integer")
        return self

class DenseLayerConfig(BaseModel):
    """Configuration for a dense (fully connected) layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        activation (str | None): Activation function to use. Options: 'relu', 'tanh', 'sigmoid'. If None, no activation is applied.
    """
    input_dim: int
    output_dim: int
    normalization: Optional[str] = None
    activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if self.activation is not None and self.activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported activation: {self.activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
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
    output_padding: Optional[int] = None
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
        layer_type (str): Type of the layer to be wrapped. Options: 'dense', 'conv', 'rnn', 'attention'.
        layer_config (BaseModel): Configuration of the layer to be wrapped with residual connection.
    """
    layer_type: str
    layer_config: BaseModel

    @model_validator(mode='after')
    def _validate(self):
        if self.layer_type.lower() not in ('dense', 'conv', 'rnn', 'attention'):
            raise ValueError(f"Unsupported layer_type: {self.layer_type}. Supported types: 'dense', 'conv', 'rnn', 'attention'")
        return self
    
class EmbeddingLayerConfig(BaseModel):
    """Configuration for an embedding layer.
    
    Args:
        num_embeddings (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        padding_idx (int | None): If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training.
    """
    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be a positive integer")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if self.padding_idx is not None and (self.padding_idx < 0 or self.padding_idx >= self.num_embeddings):
            raise ValueError("padding_idx must be a non-negative integer less than num_embeddings")
        return self
    
class PatchEmbeddingLayerConfig(BaseModel):
    """Configuration for a patch embedding layer.
    
    Args:
        embedding_dim (int): Dimension of the embeddings.
        patch_size (int): Size of each patch.
        in_channels (int): Number of input channels.
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
    
class ProjectionLayerConfig(BaseModel):
    """Configuration for a projection layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
    """
    input_dim: int
    output_dim: int

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        return self
    
class ReshapeLayerConfig(BaseModel):
    """Configuration for a reshape layer.
    
    Args:
        shape (tuple): Desired shape after reshaping.
    """
    shape: tuple

    @model_validator(mode='after')
    def _validate(self):
        if not all(isinstance(dim, int) and dim > 0 for dim in self.shape):
            raise ValueError("All dimensions in shape must be positive integers")
        return self