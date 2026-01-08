from typing import Optional
import torch
from pydantic import model_validator

from .base import BaseSample

class UniModalSample(BaseSample):
    data: Optional[torch.Tensor] = None

    @model_validator(mode="after")
    def check_data_present(self):
        if self.data is None:
            raise ValueError("The 'data' tensor must be provided for a unimodal sample.")
        return self

    def to_device(self, device: torch.device):
        super().to_device(device)
        if self.data is not None:
            self.data = self.data.to(device)
        return self

    def as_dict(self):
        result = super().as_dict()
        if self.data is not None:
            result['data'] = self.data
        return result