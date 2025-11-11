"""Recurrent Neural Networks"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..base import NeuralNetworkBase


class RecurrentNN(NeuralNetworkBase):
    """
    Basic RNN with configurable architecture.
    Base class for LSTM and GRU.
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, output_size: Optional[int] = None,
                 rnn_type: str = "rnn", bidirectional: bool = False,
                 dropout: float = 0.0):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of recurrent layers
            output_size: Output dimension (if None, uses hidden_size)
            rnn_type: Type of RNN ('rnn', 'lstm', 'gru')
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size or hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Create RNN layer
        rnn_types = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU
        }
        
        rnn_class = rnn_types.get(rnn_type.lower(), nn.RNN)
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * self.num_directions, self.output_size)
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Initial hidden state
            
        Returns:
            output: Output tensor
            hidden: Final hidden state
        """
        # RNN forward
        output, hidden = self.rnn(x, hidden)
        
        # Apply output layer to all timesteps
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state"""
        weight = next(self.parameters())
        hidden_shape = (self.num_layers * self.num_directions, 
                       batch_size, self.hidden_size)
        return weight.new_zeros(hidden_shape)


class LSTM(RecurrentNN):
    """Long Short-Term Memory network"""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, output_size: Optional[int] = None,
                 bidirectional: bool = False, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, num_layers, output_size,
                        rnn_type="lstm", bidirectional=bidirectional, dropout=dropout)
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state and cell state for LSTM"""
        weight = next(self.parameters())
        hidden_shape = (self.num_layers * self.num_directions, 
                       batch_size, self.hidden_size)
        h0 = weight.new_zeros(hidden_shape)
        c0 = weight.new_zeros(hidden_shape)
        return (h0, c0)


class GRU(RecurrentNN):
    """Gated Recurrent Unit network"""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, output_size: Optional[int] = None,
                 bidirectional: bool = False, dropout: float = 0.0):
        super().__init__(input_size, hidden_size, num_layers, output_size,
                        rnn_type="gru", bidirectional=bidirectional, dropout=dropout)