from pydantic import model_validator

from .base import BaseCoreConfig

class RecurrentUnitConfig(BaseCoreConfig):
    num_layers: int = 1
    rnn_type: str = 'RNN'
    bidirectional: bool = False

    @model_validator(mode='after')
    def _validate(self):
        if self.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if self.rnn_type.upper() not in ('LSTM', 'GRU', 'RNN'):
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}. Supported types: 'LSTM', 'GRU', 'RNN'.")
        return self