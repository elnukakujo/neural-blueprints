from typing import Optional, Dict
import torch
from pydantic import model_validator

from .base import BaseSample

class MultiModalSample(BaseSample):
    image: Optional[torch.Tensor | dict[str, torch.Tensor]] = None
    tabular: Optional[torch.Tensor | dict[str, torch.Tensor]] = None
    representation: Optional[torch.Tensor | dict[str, torch.Tensor]] = None

    @model_validator(mode="after")
    def check_at_least_one_modality(self):
        if not any([self.image, self.tabular, self.representation]):
            raise ValueError("At least one modality (image, tabular, or representation) must be provided.")
        return self

    def to_device(self, device: torch.device):
        super().to_device(device)
        for attr in ['image', 'tabular', 'representation']:
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))
        return self

    def as_dict(self):
        result = super().as_dict()
        for attr in ['image', 'tabular', 'representation']:
            tensor = getattr(self, attr)
            if tensor is not None:
                result[attr] = tensor
        return result
