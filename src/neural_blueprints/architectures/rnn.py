import torch
import torch.nn as nn

from ..components.core import RecurrentUnit
from ..config import RNNConfig
from ..utils import get_activation

class RNN(nn.Module):
    """A simple Recurrent Neural Network (RNN) architecture."""
    def __init__(self, config: RNNConfig):
        super(RNN, self).__init__()
        self.rnn = RecurrentUnit(
            config=config.rnn_unit_config
        )
        self.output_dim = config.output_dim
        self.final_activation = config.final_activation

        self.network = nn.Sequential(
            nn.Linear(config.rnn_unit_config.hidden_dim, self.output_dim),
            get_activation(self.final_activation)
        )
    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> torch.Tensor:
        rnn_out, hidden = self.rnn(x, hidden)
        return self.network(rnn_out), hidden
