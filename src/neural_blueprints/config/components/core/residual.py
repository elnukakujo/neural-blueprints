from typing import Optional
from pydantic import BaseModel, model_validator

class ResidualLayerConfig(BaseModel):
    """Configuration for a residual layer.
    
    Args:
        layer_config (BaseModel): Configuration of the layer to be wrapped with residual connection.
        normalization (str | None): Configuration for normalization layer. If None, no normalization is applied.
        activation (str | None): Activation function to use. If None, no activation is applied.
        dropout_p (float | None): Dropout probability. If None, no dropout is applied.
    """
    layer_config: BaseModel
    normalization: Optional[str] = None
    activation: Optional[str] = None
    dropout_p: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.layer_type.lower() not in ('dense', 'conv', 'rnn', 'attention'):
            raise ValueError(f"Unsupported layer_type: {self.layer_type}. Supported types: 'dense', 'conv', 'rnn', 'attention'")
        return self