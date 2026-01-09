import torch
import torch.nn as nn

from ..components.core import RecurrentUnit
from ..config.architectures import RNNConfig
from ..utils import get_activation

class RNN(nn.Module):
    """A simple Recurrent Neural Network (RNN) architecture.

    Args:
        config (RNNConfig): Configuration for the RNN model.
    """
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
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the RNN.

        Args:
            x (torch.Tensor): Input tensor.
            hidden (torch.Tensor, optional): Hidden state tensor. Defaults to None.
            
        Returns:
            Output tensor after passing through the RNN and the updated hidden state.
        """
        rnn_out, hidden = self.rnn(x, hidden)
        return self.network(rnn_out), hidden
