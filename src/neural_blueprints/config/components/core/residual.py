from pydantic import BaseModel, model_validator

class ResidualLayerConfig(BaseModel):
    """Configuration for a residual layer.
    
    Args:
        layer_type (str): Type of the layer to be wrapped. Options: 'dense', 'conv', 'rnn', 'attention'.
        layer_config (BaseModel): Configuration of the layer to be wrapped with residual connection.
    """
    layer_type: str
    layer_config: BaseModel

    @model_validator(mode='after')
    def _validate(self):
        if self.layer_type.lower() not in ('dense', 'conv', 'rnn', 'attention'):
            raise ValueError(f"Unsupported layer_type: {self.layer_type}. Supported types: 'dense', 'conv', 'rnn', 'attention'")
        return self