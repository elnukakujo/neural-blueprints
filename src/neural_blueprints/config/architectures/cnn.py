from typing import List, Optional
from pydantic import BaseModel, model_validator

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