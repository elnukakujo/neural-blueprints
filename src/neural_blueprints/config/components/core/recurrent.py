from typing import Optional
from pydantic import BaseModel, model_validator

class RecurrentUnitConfig(BaseModel):
    """Configuration for a recurrent neural network unit.
    
    Args:
        rnn_type (str): Type of RNN. Options: 'LSTM', 'GRU', 'RNN'.
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        bidirectional (bool): If True, becomes a bidirectional RNN.
        normalization (str | None): Configuration for normalization layer. If None, no normalization is applied.
        activation (str | None): Activation function to use. If None, no activation is applied.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
    """
    input_dim: int
    hidden_dim: int
    num_layers: int = 1
    rnn_type: str = 'RNN'
    bidirectional: bool = False
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if self.rnn_type.upper() not in ('LSTM', 'GRU', 'RNN'):
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}. Supported types: 'LSTM', 'GRU', 'RNN'.")
        return self