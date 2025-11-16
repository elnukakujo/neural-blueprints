from typing import List, Tuple, Optional, Any, Dict
from pydantic import BaseModel, model_validator, Field
import numpy as np

from ..config.core import NormalizationConfig
from ..config.composite import GeneratorConfig, DiscriminatorConfig, TransformerDecoderConfig, TransformerEncoderConfig

class MLPConfig(BaseModel):
    """Configuration for a Multi-Layer Perceptron (MLP) architecture."""

    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    normalization: Optional[str] = None
    activation: Optional[str] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if any(h <= 0 for h in self.hidden_dims):
            raise ValueError("All hidden_dims must be positive integers")
        if self.activation is not None and self.activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu'):
            raise ValueError(f"Unsupported activation: {self.activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu'}")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu'}")
        return self
    
class CNNConfig(BaseModel):
    """Configuration for a Convolutional Neural Network (CNN) architecture."""

    layer_types: List[str]
    layer_configs: List[BaseModel]
    feedforward_config: BaseModel
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu', 'softmax'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu', 'softmax'}")
        for layer_type in self.layer_types:
            if layer_type.lower() not in ('conv1d', 'conv2d', 'pool1d', 'pool2d', 'flatten'):
                raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {'conv1d', 'conv2d', 'pool1d', 'pool2d', 'flatten'}")
        return self
    
class RNNConfig(BaseModel):
    """Configuration for a Recurrent Neural Network (RNN) architecture."""

    rnn_unit_config: BaseModel
    output_dim: int
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu'}")
        return self
    
class AutoEncoderConfig(BaseModel):
    """Configuration for an AutoEncoder architecture."""

    encoder_layer_types: List[str]
    encoder_layer_configs: List[BaseModel]
    decoder_layer_types: List[str]
    decoder_layer_configs: List[BaseModel]

    @model_validator(mode='after')
    def _validate(self):
        if len(self.encoder_layer_types) != len(self.encoder_layer_configs):
            raise ValueError("Length of encoder_layer_types must match length of encoder_layer_configs")
        if len(self.decoder_layer_types) != len(self.decoder_layer_configs):
            raise ValueError("Length of decoder_layer_types must match length of decoder_layer_configs")
        return self
    
class VariationalAutoEncoderConfig(AutoEncoderConfig):
    """Configuration for a Variational AutoEncoder (VAE) architecture."""

    reconstruction_factor: float = Field(1.0, ge=0.0)
    kl_divergence_factor: float = Field(1.0, ge=0.0)

    @model_validator(mode='after')
    def _validate(self):
        if self.reconstruction_factor < 0.0:
            raise ValueError("reconstruction_factor must be non-negative")
        if self.kl_divergence_factor < 0.0:
            raise ValueError("kl_divergence_factor must be non-negative")

        return self
    
class GANConfig(BaseModel):
    """Configuration for a Generative Adversarial Network (GAN) architecture."""

    generator_config: GeneratorConfig
    discriminator_config: DiscriminatorConfig

    @model_validator(mode='after')
    def _validate(self):
        return self

class TransformerConfig(BaseModel):
    """Configuration for a Transformer architecture."""

    encoder_config: TransformerEncoderConfig
    decoder_config: TransformerDecoderConfig
    output_dim: Optional[int] = None

    @model_validator(mode='after')
    def _validate(self):
        return self
    
class TabularBERTConfig(BaseModel):
    """Configuration for a BERT-style architecture."""

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