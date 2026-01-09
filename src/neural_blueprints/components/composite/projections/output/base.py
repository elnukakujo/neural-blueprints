from pydantic import model_validator
from ... import BaseComposite

class BaseOutputProjection(BaseComposite):
    @model_validator(mode='after')
    def _validate(self):
        if len(self.input_dim) != 2:
            raise ValueError("Input dimension must be of length 2 (num_features, latent_dim).")