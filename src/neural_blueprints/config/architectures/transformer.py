from typing import Optional
from pydantic import BaseModel, model_validator

from ..core import NormalizationConfig
from ..composite import TransformerDecoderConfig, TransformerEncoderConfig

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
        cardinalities (List[int]): List of cardinalities for each feature (1 for continuous).
        encoder_config (TransformerEncoderConfig): Configuration for the transformer encoder.
        dropout (float): Dropout rate between 0.0 and 1.0.
        with_input_projection (bool): Whether to use input projections for each feature.
        with_output_projection (bool): Whether to use output projections for masked attribute prediction.
        final_normalization (Optional[NormalizationConfig]): Normalization to apply after the final layer.
        final_activation (Optional[str]): Activation function to apply after the final layer.
    """

    cardinalities: list[int]
    encoder_config: TransformerEncoderConfig
    dropout: float
    with_input_projection: bool = True
    with_output_projection: bool = True
    final_normalization: Optional[NormalizationConfig] = None
    final_activation: Optional[str] = None


    @model_validator(mode='after')
    def _validate(self):
        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
        if self.final_normalization is not None and self.final_normalization.norm_type.lower() not in ('batchnorm1d', 'batchnorm2d', 'layernorm'):
            raise ValueError(f"Unsupported final_normalization: {self.final_normalization.norm_type}. Supported: {'batchnorm1d', 'batchnorm2d', 'layernorm'}")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self