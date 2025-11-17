from typing import List, Tuple, Optional, Any, Dict
from pydantic import BaseModel, model_validator, Field
import numpy as np

from .core import NormalizationConfig
from .composite import GeneratorConfig, DiscriminatorConfig, TransformerDecoderConfig, TransformerEncoderConfig

class MLPConfig(BaseModel):
    """Configuration for a Multi-Layer Perceptron (MLP) architecture.
    
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dims (List[int]): List of hidden layer dimensions.
        output_dim (int): Dimension of the output features.
        normalization (Optional[str]): Normalization to apply after each layer.
        activation (Optional[str]): Activation function to apply after each layer.
        final_activation (Optional[str]): Activation function to apply after the final layer.
    """

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
    """Configuration for a Convolutional Neural Network (CNN) architecture.
    
    Args:
        layer_types (list of str): List of layer types in the CNN.
        layer_configs (list of BaseModel): List of configurations for each layer.
        feedforward_config (BaseModel): Configuration for the feedforward network after convolutional layers.
        final_activation (Optional[str]): Activation function to apply after the final layer.
    """

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
    """Configuration for a Recurrent Neural Network (RNN) architecture.
    
    Args:
        rnn_unit_config (BaseModel): Configuration for the RNN unit (e.g. RNN, LSTM, GRU) including parameters like hidden size, number of layers, etc.
        output_dim (int): Dimension of the output features.
        final_activation (Optional[str]): Activation function to apply to the output.
    """

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
    """Configuration for an AutoEncoder architecture.
    
    Args:
        encoder_layer_types (list of str): List of layer types for the encoder.
        encoder_layer_configs (list of BaseModel): List of configurations for each encoder layer.
        decoder_layer_types (list of str): List of layer types for the decoder.
        decoder_layer_configs (list of BaseModel): List of configurations for each decoder layer.
    """

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
    
class GANConfig(BaseModel):
    """Configuration for a Generative Adversarial Network (GAN) architecture.
    
    Args:
        generator_config (GeneratorConfig): Configuration for the generator model.
        discriminator_config (DiscriminatorConfig): Configuration for the discriminator model.
    """

    generator_config: GeneratorConfig
    discriminator_config: DiscriminatorConfig

    @model_validator(mode='after')
    def _validate(self):
        return self

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