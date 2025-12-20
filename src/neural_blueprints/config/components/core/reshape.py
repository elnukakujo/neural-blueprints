from pydantic import BaseModel, model_validator
    
class ReshapeLayerConfig(BaseModel):
    """Configuration for a reshape layer.
    
    Args:
        shape (tuple): Desired shape after reshaping.
    """
    shape: tuple

    @model_validator(mode='after')
    def _validate(self):
        if not all(isinstance(dim, int) and dim > 0 for dim in self.shape):
            raise ValueError("All dimensions in shape must be positive integers")
        return self