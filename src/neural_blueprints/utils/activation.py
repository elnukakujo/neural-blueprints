import torch.nn as nn
from typing import Optional

def get_activation(activation_name: Optional[str] = None) -> nn.Module:
    """Returns the activation function corresponding to the given name."""
    if activation_name is None:
        return nn.Identity()
    activation_name = activation_name.lower()
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name == 'softmax':
        return nn.Softmax(dim=-1)
    elif activation_name == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation_name == 'elu':
        return nn.ELU()
    elif activation_name == 'silu':
        return nn.SiLU()
    elif activation_name == 'relu6':
        return nn.ReLU6()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")