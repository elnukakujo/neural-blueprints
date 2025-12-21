import torch
import torch.nn as nn
from pydantic import BaseModel

from .layer_type import infer_layer_type

def get_block(block_config: BaseModel) -> nn.Module:
    """Factory function to create neural network blocks based on the block type.
    
    Args:
        block_config (BaseModel): Configuration object for the block.
        
    Returns:
        nn.Module: An instance of the specified block type.
    """
    block_type = infer_layer_type(block_config)
    if block_type == 'dense':
        from ..components.core import DenseLayer
        return DenseLayer(block_config)
    elif block_type == 'conv1d':
        from ..components.core import Conv1dLayer
        return Conv1dLayer(block_config)
    elif block_type == 'conv2d':
        from ..components.core import Conv2dLayer
        return Conv2dLayer(block_config)
    elif block_type == 'conv3d':
        from ..components.core import Conv3dLayer
        return Conv3dLayer(block_config)
    elif block_type == 'conv1d_transpose':
        from ..components.core import Conv1dTransposeLayer
        return Conv1dTransposeLayer(block_config)
    elif block_type == 'conv2d_transpose':
        from ..components.core import Conv2dTransposeLayer
        return Conv2dTransposeLayer(block_config)
    elif block_type == 'conv3d_transpose':
        from ..components.core import Conv3dTransposeLayer
        return Conv3dTransposeLayer(block_config)
    elif block_type == 'attention':
        from ..components.core import AttentionLayer
        return AttentionLayer(block_config)
    elif block_type == 'recurrent':
        from ..components.core import RecurrentUnit
        return RecurrentUnit(block_config)
    elif block_type == 'embedding':
        from ..components.core import EmbeddingLayer
        return EmbeddingLayer(block_config)
    elif block_type == 'residual':
        from ..components.core import ResidualLayer
        return ResidualLayer(block_config)
    elif block_type == 'patch_embedding':
        from ..components.core import PatchEmbeddingLayer
        return PatchEmbeddingLayer(block_config)
    elif block_type == 'pool1d':
        from ..components.core import Pooling1dLayer
        return Pooling1dLayer(block_config)
    elif block_type == 'pool2d':
        from ..components.core import Pooling2dLayer
        return Pooling2dLayer(block_config)
    elif block_type == 'feedforward':
        from ..components.composite import FeedForwardNetwork
        return FeedForwardNetwork(block_config)
    elif block_type == 'encoder':
        from ..components.composite import Encoder
        return Encoder(block_config)
    elif block_type == 'decoder':
        from ..components.composite import Decoder
        return Decoder(block_config)
    elif block_type == 'generator':
        from ..components.composite import Generator
        return Generator(block_config)
    elif block_type == 'discriminator':
        from ..components.composite import Discriminator
        return Discriminator(block_config)
    elif block_type == 'flatten':
        from ..components.core import FlattenLayer
        return FlattenLayer(block_config)
    elif block_type == 'reshape':
        from ..components.core import ReshapeLayer
        return ReshapeLayer(block_config)
    elif block_type == 'norm':
        from ..components.core import NormalizationLayer
        return NormalizationLayer(block_config)
    else:
        raise ValueError(f"Unsupported block type: {block_type}")