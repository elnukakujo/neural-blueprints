import torch
import torch.nn as nn

from ..components.core import RecurrentUnit
from ..config import RNNConfig
from ..utils import get_activation

class RNN(nn.Module):
    """A simple Recurrent Neural Network (RNN) architecture."""
    def __init__(self, config: RNNConfig):
        super(RNN, self).__init__()
        self.config = config

        self.rnn = RecurrentUnit(
            config=config.rnn_unit_config
        )

        self.network = nn.Sequential(
            nn.Linear(config.rnn_unit_config.hidden_dim, config.output_dim),
            get_activation(config.final_activation)
        )

    def blueprint(self) -> RNNConfig:
        print(self)
        return self.config
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> torch.Tensor:
        rnn_out, hidden = self.rnn(x, hidden)
        return self.network(rnn_out), hidden
