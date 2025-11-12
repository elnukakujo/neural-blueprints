"""Neural Network Backbones Library"""
from .activation import get_activation
from .blocks import get_block

__version__ = "0.1.0"
__all__ = [
    get_activation,
    get_block,
]