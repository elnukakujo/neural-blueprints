import torch
import torch.nn as nn

from ..core import DenseLayer, Conv2dTransposeLayer, Conv1dTransposeLayer, AttentionLayer
from ...config import DecoderConfig

class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super(Decoder, self).__init__()

        self.layer_types = config.layer_types
        self.layer_configs = config.layer_configs
        
        self.layers = nn.ModuleList()
        
        for layer_type, layer_config in zip(self.layer_types, self.layer_configs):
            if layer_type == 'dense':
                self.layers.append(DenseLayer(layer_config))
            elif layer_type == 'conv1d_transpose':
                self.layers.append(Conv1dTransposeLayer(layer_config))
            elif layer_type == 'conv2d_transpose':
                self.layers.append(Conv2dTransposeLayer(layer_config))
            elif layer_type == 'attention':
                self.layers.append(AttentionLayer(layer_config))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        
        self.network = nn.Sequential(*self.layers)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)