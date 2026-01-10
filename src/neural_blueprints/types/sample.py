import torch
from typing import Optional

from .base import BaseType
from .modalities import Modalities

class Sample(BaseType, total=False):
    """
    Type definition for dataset batches.

    label (Optional[torch.Tensor | dict[str, torch.Tensor]]): The labels associated with the data.
    metadata (Optional[dict[str, torch.Tensor]]): Additional metadata for the sample like a mask.
    """
    label: Optional[torch.Tensor | dict[str, torch.Tensor]]
    metadata: Optional[dict[str, torch.Tensor]]

class UniModalSample(Sample):
    """
    Type definition for unimodal dataset samples.
    
    Required:
        - input: The single input tensor
    
    Optional (inherited from Sample):
        - label: Labels (can be single tensor or dict)
        - metadata: Additional metadata tensors
    """
    inputs: torch.Tensor

class MultiModalSample(Sample):
    """
    Type definition for multimodal dataset samples.

    Required:
        - input (Modalities): A dict containing tensors for at least one modality.
            - tabular (Optional[torch.Tensor | dict[str, torch.Tensor]]): Tabular/structured data
            - image (Optional[torch.Tensor | dict[str, torch.Tensor]]): Image data
            - representation (Optional[torch.Tensor | dict[str, torch.Tensor]]): Pre-computed embeddings/representations
            - audio (Optional[torch.Tensor | dict[str, torch.Tensor]]): Audio data
            - text (Optional[torch.Tensor | dict[str, torch.Tensor]]): Text data
    
    Optional (inherited from Sample):
        - label: The labels associated with the data (can be single tensor or dict)
        - metadata: Additional metadata for the sample like masks for each modality
    """
    inputs: Modalities