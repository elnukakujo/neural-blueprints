from typing import Optional
from pydantic import BaseModel, model_validator
    
class EmbeddingLayerConfig(BaseModel):
    """Configuration for an embedding layer.
    
    Args:
        num_embeddings (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        padding_idx (int | None): If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training.
        normalization (str | None): Normalization method to apply after embedding.
        activation (str | None): Activation function to apply after embedding.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
    """
    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int] = 0
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None

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
        normalization (str | None): Normalization method to apply after embedding.
        activation (str | None): Activation function to apply after embedding.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
    """
    embedding_dim: int
    patch_size: int
    in_channels: int = 3
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        return self