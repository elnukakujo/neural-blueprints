from typing import Optional, Dict, Any
import torch
from pydantic import BaseModel

class BaseSample(BaseModel):
    """
    Base class for dataset samples (unimodal or multimodal).
    
    Common attributes:
        label: Optional target tensor
        sample_id: Optional identifier for the sample
        metadata: Optional dictionary for additional info
    """
    label: Optional[torch.Tensor | dict[str, torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_device(self, device: torch.device):
        """
        Move all tensors in this sample to the given device.
        Override in subclasses to include modality-specific tensors.
        """
        if self.label is not None:
            if isinstance(self.label, dict):
                self.label = {k: v.to(device) for k, v in self.label.items()}
            else:
                self.label = self.label.to(device)
        return self

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return all non-None tensors in the sample.
        Override in subclasses for modality-specific tensors.
        """
        result = {}
        if self.label is not None:
            result['label'] = self.label
        return result
    
    def get_mask(self):
        """
        Retrieve mask from metadata if available.
        """
        return self.metadata.get("mask") if self.metadata else None