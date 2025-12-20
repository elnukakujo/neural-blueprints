from pydantic import BaseModel, model_validator

class ProjectionLayerConfig(BaseModel):
    """Configuration for a projection layer.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
    """
    input_dim: int
    output_dim: int

    @model_validator(mode='after')
    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        return self