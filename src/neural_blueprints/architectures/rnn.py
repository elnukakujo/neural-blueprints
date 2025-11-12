import torch
import torch.nn as nn

from ..components.core import RecurrentUnit
from ..config import RNNConfig

class RNN(nn.Module):
    """A simple Recurrent Neural Network (RNN) architecture."""
    def __init__(self, config: RNNConfig):
        super(RNN, self).__init__()
        self.rnn = RecurrentUnit(
            config=config.rnn_unit_config
        )
        self.output_dim = config.output_dim
        self.final_activation = config.final_activation
