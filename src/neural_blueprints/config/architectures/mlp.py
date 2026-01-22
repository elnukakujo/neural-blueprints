from pydantic import model_validator

from .base import BaseArchitectureConfig

class MLPConfig(BaseArchitectureConfig):
    hidden_dims: list[int]
    @model_validator(mode='after')
    def _validate(self):
        BaseArchitectureConfig._validate(self)
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'gelu', 'softmax'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'gelu', 'softmax'}")
        return self