import torch
from torch.utils.data import Dataset
from typing import TypedDict, Optional
from abc import ABC, abstractmethod

class Sample(TypedDict, total=False):
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
        - data: The main data tensor
    
    Optional (inherited from Sample):
        - label: Labels (can be single tensor or dict)
        - metadata: Additional metadata tensors
    """
    data: torch.Tensor

class MultiModalSample(Sample):
    """
    Type definition for multimodal dataset samples.
    
    At least one modality should be present (tabular, image, representation, audio, text).
    
    Optional modalities:
        - tabular: Tabular/structured data
        - image: Image data
        - representation: Pre-computed embeddings/representations
        - audio: Audio data
        - text: Text data
    
    Optional (inherited from Sample):
        - label: Labels
        - metadata: Additional metadata
    """

    # The different modalities
    tabular: Optional[torch.Tensor | dict[str, torch.Tensor]]
    image: Optional[torch.Tensor | dict[str, torch.Tensor]]
    representation: Optional[torch.Tensor | dict[str, torch.Tensor]]
    audio: Optional[torch.Tensor | dict[str, torch.Tensor]]
    text: Optional[torch.Tensor | dict[str, torch.Tensor]]

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.
    
    Cannot be instantiated directly - must subclass and implement:
        - __init__: Initialize the dataset
        - __len__: Return dataset size
        - __getitem__: Return a sample
    """
    def __init__(self):
        raise NotImplementedError("Subclasses must implement __init__ method.")
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__ method.")

    @abstractmethod
    def __getitem__(self, idx) -> UniModalSample | MultiModalSample:
        """
        Return a sample at the given index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Either a UniModalSample or MultiModalSample dict
        """
        raise NotImplementedError("Subclasses must implement __getitem__ method.")