from typing import Optional
from pydantic import BaseModel, model_validator

class RNNConfig(BaseModel):
    """Configuration for a Recurrent Neural Network (RNN) architecture.
    
    Args:
        rnn_unit_config (BaseModel): Configuration for the RNN unit (e.g. RNN, LSTM, GRU) including parameters like hidden size, number of layers, etc.
        output_dim (int): Dimension of the output features.
        final_activation (Optional[str]): Activation function to apply to the output.
    """

    rnn_unit_config: BaseModel
    output_dim: int
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu'}")
        return self