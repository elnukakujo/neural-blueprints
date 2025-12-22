from typing import Optional, List
from pydantic import BaseModel, model_validator

from ..components.core import NormalizationLayerConfig
from ..components.composite import TransformerDecoderConfig, TransformerEncoderConfig

class TransformerConfig(BaseModel):
    """Configuration for a Transformer architecture.
    
    Args:
        encoder_config (TransformerEncoderConfig): Configuration for the transformer encoder.
        decoder_config (TransformerDecoderConfig): Configuration for the transformer decoder.
        output_dim (Optional[int]): Dimension of the output features. If None, uses decoder hidden_dim.
    """

    encoder_config: TransformerEncoderConfig
    decoder_config: TransformerDecoderConfig
    output_dim: Optional[int] = None

    @model_validator(mode='after')
    def _validate(self):
        return self
    
class TabularBERTConfig(BaseModel):
    """Configuration for a BERT-style architecture.
    
    Args:
        input_cardinalities (List[int]): List of cardinalities for each input categorical features.
        output_cardinalities (Optional[List[int]]): List of cardinalities for each output categorical features. If None, use input_cardinalities.
        latent_dim (int): Dimension of the latent embeddings.
        dropout_p (Optional[float]): Dropout probability. If None, no dropout is applied.
        normalization (Optional[str]): Normalization layer type. If None, no normalization is applied.
        activation (Optional[str]): Activation function type. If None, no activation is applied.
        final_activation (Optional[str]): Final activation function type. If None, no final activation is applied.
    """
    input_cardinalities: List[int]
    output_cardinalities: Optional[List[int]] = None
    latent_dim: int
    encoder_layers: int
    dropout_p: Optional[float] = None
    normalization: Optional[str] = None
    activation: Optional[str] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.output_cardinalities is None:
            self.output_cardinalities = self.input_cardinalities
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self