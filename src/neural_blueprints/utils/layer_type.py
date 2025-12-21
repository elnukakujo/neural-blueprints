from typing import List
from pydantic import BaseModel

from ..config.components.core import (
    ConvLayerConfig,
    DenseLayerConfig,
    FlattenLayerConfig,
    ReshapeLayerConfig,
    NormalizationLayerConfig,
)

def infer_layer_type(config: BaseModel) -> str:
    """
    Infer the layer type from a layer configuration object.
    
    Args:
        config: A layer configuration object
        
    Returns:
        A string representing the layer type
    """
    type_mapping = {
        ConvLayerConfig: 'conv2d',
        DenseLayerConfig: 'dense',
        FlattenLayerConfig: 'flatten',
        ReshapeLayerConfig: 'reshape',
        NormalizationLayerConfig: 'norm'
    }
        
    # Special case: check if it's a transposed convolution
    if type(config) == ConvLayerConfig:
        if config.dim == 1:
            layer_type = 'conv1d'
        elif config.dim == 2:
            layer_type = 'conv2d'
        elif config.dim == 3:
            layer_type = 'conv3d'
        else:
            raise ValueError(f"Unsupported convolution dimension: {config.dim}")
        
        if config.in_channels > config.out_channels:
            layer_type += '_transpose'
        return layer_type
    
    return type_mapping.get(type(config), 'unknown')