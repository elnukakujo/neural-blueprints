import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from ..types import UniModalSample, MultiModalSample
    
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
    
    def random_split(self, lengths: list[float] | list[int]) -> list['BaseDataset']:
        """
        Split the dataset into non-overlapping new datasets of given lengths.
        
        Args:
            lengths (list[float] | list[int]): If floats, they represent proportions summing to 1.0.
                                               If ints, they represent absolute sizes.

        Returns:
            List of BaseDataset instances representing the splits.
        """
        total_length = len(self)
        if all(isinstance(length, float) for length in lengths):
            assert abs(sum(lengths) - 1.0) < 1e-6, "Proportions must sum to 1.0"
            lengths = [int(length * total_length) for length in lengths]
            # Adjust last length to account for rounding errors
            lengths[-1] = total_length - sum(lengths[:-1])
        elif all(isinstance(length, int) for length in lengths):
            assert sum(lengths) == total_length, "Sum of lengths must equal dataset size"
        else:
            raise ValueError("Lengths must be all floats or all integers.")
        
        return torch.utils.data.random_split(self, lengths)