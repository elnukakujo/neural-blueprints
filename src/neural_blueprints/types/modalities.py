import torch
from typing import Optional

from .base import BaseType

class Modalities(BaseType):
    """
    Type definition for different data modalities in a multimodal dataset.
    
    Optional modalities:
        - tabular (Optional[torch.Tensor | dict[str, torch.Tensor]]): Tabular/structured data
        - image (Optional[torch.Tensor | dict[str, torch.Tensor]]): Image data
        - representation (Optional[torch.Tensor | dict[str, torch.Tensor]]): Pre-computed embeddings/representations
        - audio (Optional[torch.Tensor | dict[str, torch.Tensor]]): Audio data
        - text (Optional[torch.Tensor | dict[str, torch.Tensor]]): Text data
    """
    tabular: Optional[torch.Tensor | dict[str, torch.Tensor]]
    image: Optional[torch.Tensor | dict[str, torch.Tensor]]
    representation: Optional[torch.Tensor | dict[str, torch.Tensor]]
    audio: Optional[torch.Tensor | dict[str, torch.Tensor]]
    text: Optional[torch.Tensor | dict[str, torch.Tensor]]