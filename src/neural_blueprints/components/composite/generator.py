import torch
import torch.nn as nn

from .decoder import Decoder
from ..core import DenseLayer, Conv2dTransposeLayer, Conv1dTransposeLayer, AttentionLayer
from ...config import GeneratorConfig

class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super(Generator, self).__init__()

        self.latent_dim = config.latent_dim
        self.output_shape = config.output_shape
        self.architecture = config.architecture
        
        self.layers = nn.ModuleList()
        
        