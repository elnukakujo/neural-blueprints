import torch
import torch.nn as nn
from ...config import RecurrentUnitConfig

class RecurrentUnit(nn.Module):
    def __init__(self, config: RecurrentUnitConfig):
        super(RecurrentUnit, self).__init__()
        
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.rnn_type = config.rnn_type.upper()

        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        else:
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}. Supported types: 'LSTM', 'GRU', 'RNN'.")

    def forward(self, x: torch.Tensor, hidden=None) -> tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.rnn(x, hidden)
        return output, hidden