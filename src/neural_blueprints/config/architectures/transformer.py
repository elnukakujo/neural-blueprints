from typing import Optional, List
from pydantic import BaseModel, model_validator

from ..components.composite import TransformerDecoderConfig, TransformerEncoderConfig
from ..components.composite.projections.output import BaseProjectionOutputConfig

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
        input_cardinalities (List[int]): List of cardinalities for each input categorical feature.
        output_cardinalities (Optional[List[int]]): List of cardinalities for each output categorical feature. 
            If None, defaults to input_cardinalities.
        output_projection (Optional[BaseProjectionOutputConfig]): Configuration for the output projection layer.
        latent_dim (int): Dimension of the latent embeddings.
        encoder_layers (int): Number of transformer encoder layers.
        dropout_p (Optional[float]): Dropout probability. If None, no dropout is applied.
        normalization (Optional[str]): Type of normalization to use ('layernorm', 'batchnorm', etc.). If None, no normalization is applied.
        activation (Optional[str]): Activation function to use ('relu', 'gelu', etc.). If None, no activation is applied.
        final_activation (Optional[str]): Activation function to apply at the final output layer. 
            If None, no final activation is applied.
    """
    input_cardinalities: List[int]
    output_cardinalities: Optional[List[int]] = None
    output_projection: Optional[BaseProjectionOutputConfig] = None
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