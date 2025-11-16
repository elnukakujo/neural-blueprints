import torch
import torch.nn as nn
from typing import Optional

from ..config import NormalizationConfig

def get_normalization(config: Optional[NormalizationConfig]) -> nn.Module:
    """Returns a normalization layer based on the specified type.

    Args:
        norm_type (str | None): Type of normalization. Options: 'batchnorm', 'layernorm', 'instancenorm', 'groupnorm'. If None, no normalization is applied.
        num_features (int): Number of features/channels for the normalization layer.

    Returns:
        nn.Module | None: The corresponding normalization layer or None if norm_type is None.
    """
    if config is None or config.norm_type is None:
        return nn.Identity()

    norm_type = config.norm_type.lower()
    if norm_type == 'batchnorm1d':
        return nn.BatchNorm1d(config.num_features)
    elif norm_type == 'batchnorm2d':
        return nn.BatchNorm2d(config.num_features)
    elif norm_type == 'layernorm':
        return nn.LayerNorm(config.num_features)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}. Supported types: 'batchnorm1d', 'batchnorm2d', 'layernorm'")