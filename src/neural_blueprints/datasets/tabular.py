import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset

from ..utils import get_device

class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data, handling both discrete and continuous features.

    Args:
        data (pd.DataFrame): The tabular data.
        discrete_features (list[str]): List of column names for discrete features.
        continuous_features (list[str]): List of column names for continuous features.
        device (str, optional): Device to load the data onto. Defaults to None, which uses get_device().
    """
    def __init__(
            self, 
            data: pd.DataFrame,
            discrete_features: list[str],
            continuous_features: list[str],
            device: str = None
        ):
        # Convert to tensor
        device = get_device() if device is None else device
        self.data = torch.tensor(data.to_numpy(), dtype=torch.float32, device=device)
        self.columns = data.columns.tolist()

        self.discrete_features = discrete_features
        self.continuous_features = continuous_features
        
        # Build cardinalities and token indices
        self.cardinalities = []
        self.nan_tokens = []
        
        for col in self.columns:
            if col in discrete_features:
                # Categorical: cardinality is number of unique values
                self.cardinalities.append(int(data[col].nunique()))
                self.nan_tokens.append(float(0))        # 0 for discrete
            elif col in continuous_features:
                # Continuous
                self.cardinalities.append(1)
                self.nan_tokens.append(float(-1))       # -1 for continuous
            else:
                raise ValueError(f"Column {col} not found in discrete or continuous features.")
        self.nan_tokens = torch.tensor(self.nan_tokens, dtype=self.data.dtype, device=device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        return row
    
class TabularSingleLabelDataset(TabularDataset):
    """
    PyTorch Dataset for tabular data with single label classification/regression.

    Args:
        data (pd.DataFrame): The tabular data.
        discrete_features (list[str]): List of column names for discrete features.
        continuous_features (list[str]): List of column names for continuous features.
        label_column (str): The column name for the classification/regression labels.
        device (str, optional): Device to load the data onto. Defaults to None, which uses get_device().
    """
    def __init__(
            self,
            data: pd.DataFrame,
            discrete_features: list[str],
            continuous_features: list[str],
            label_column: str,
            device: str = None
        ):
        super().__init__(data, discrete_features, continuous_features, device=device)
        self.label_idx = self.columns.index(label_column)
        self.labels = self.data[:, self.label_idx].long()

        self.cardinalities.pop(self.label_idx)
        self.data = self.data[:, torch.arange(self.data.shape[1]) != self.label_idx]
    
    def __getitem__(self, idx):
        row = self.data[idx]
        label = self.labels[idx]
        return row, label

class MaskedTabularDataset(TabularDataset):
    """
    PyTorch Dataset for tabular data with masking for self-supervised learning.

    Args:
        data (pd.DataFrame): The tabular data.
        discrete_features (list[str]): List of column names for discrete features.
        continuous_features (list[str]): List of column names for continuous features.
        mask_prob (float): Probability of masking individual values. Default is 0.1.
        mask_column (str, optional): If provided, masks the entire specified column for all samples.
        device (str, optional): Device to load the data onto. Defaults to None, which uses get_device().
    """
    def __init__(
            self,
            data: pd.DataFrame,
            discrete_features: list[str],
            continuous_features: list[str],
            mask_prob: float = 0.1,
            mask_column: str = None,
            device: str = None
        ):
        super().__init__(data, discrete_features, continuous_features, device=device)
        
        if mask_column is not None:
            # Mask entire column for all samples
            self.mask_column = mask_column
            self.col_idx = self.columns.index(mask_column)
            self.mask = torch.zeros(self.data.shape[0], self.data.shape[1], dtype=torch.bool, device=self.data.device)
            self.mask[:, self.col_idx] = True
        else:
            # Randomly mask individual values across all samples and attributes
            gen = torch.Generator(device=self.data.device)
            if os.getenv("RANDOM_SEED") is not None:
                gen = gen.manual_seed(int(os.getenv("RANDOM_SEED")))
            self.mask = torch.rand(
                self.data.shape[0], 
                self.data.shape[1], 
                generator=gen,
                device=self.data.device
            ) < mask_prob
        
        self.masked_data = self.data.clone()
        self.labels = self.data.clone()
        
        # Convert nan_tokens list to tensor and broadcast to match data shape
        # (n_attributes,) -> (1, n_attributes) -> (n_samples, n_attributes)
        nan_tokens_expanded = self.nan_tokens.unsqueeze(0).expand_as(self.data)
        
        # Apply masking using torch.where
        # Where mask is True: use nan_token, otherwise keep original value
        self.masked_data = torch.where(self.mask, nan_tokens_expanded, self.masked_data)

        # For labels: where mask is False (not masked), use nan_token
        self.labels = torch.where(~self.mask, nan_tokens_expanded, self.labels)
    
    def __getitem__(self, idx):
        row = self.masked_data[idx]
        labels = self.labels[idx]
        mask = self.mask[idx]
        
        return row, labels, mask
    
class FeatureTabularDataset(TabularDataset):
    """
    PyTorch Dataset for tabular data that includes feature representations generated by a representation model.

    Args:
        data (pd.DataFrame): The tabular data.
        discrete_features (list[str]): List of column names for discrete features.
        continuous_features (list[str]): List of column names for continuous features.
        representation_model (nn.Module): The representation model used to generate feature embeddings.
        device (str, optional): Device to load the data onto. Defaults to None, which uses get_device().
    """
    def __init__(
            self,
            data: pd.DataFrame,
            discrete_features: list[str],
            continuous_features: list[str],
            representation_model: nn.Module,
            device: str = None
        ):
        super().__init__(data, discrete_features, continuous_features, device=device)
        self.embedding_model = representation_model.to(get_device() if device is None else device)

        self.feature_data = self.embedding_model.encode(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        embedded_row = self.feature_data[idx]
        return row, embedded_row
    
class FeatureMaskedTabularDataset(MaskedTabularDataset):
    """
    PyTorch Dataset for masked tabular data that includes feature representations generated by a representation model.

    Args:
        data (pd.DataFrame): The tabular data.
        discrete_features (list[str]): List of column names for discrete features.
        continuous_features (list[str]): List of column names for continuous features.
        representation_model (RepresentationModelBase): The representation model used to generate feature embeddings.
        mask_prob (float): Probability of masking individual values. Default is 0.15.
        mask_column (str, optional): If provided, masks the entire specified column for all samples
        device (str, optional): Device to load the data onto. Defaults to None, which uses get_device().
    """
    def __init__(
            self,
            data: pd.DataFrame,
            discrete_features: list[str],
            continuous_features: list[str],
            representation_model: nn.Module,
            mask_prob: float = 0.15,
            mask_column: str = None,
            device: str = None
        ):
        super().__init__(data, discrete_features, continuous_features, mask_prob, mask_column, device=device)
        self.embedding_model = representation_model.to(get_device() if device is None else device)

        self.masked_feature_data = self.embedding_model.encode(self.masked_data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        masked_embedded_row = self.masked_feature_data[idx]
        labels = self.labels[idx]
        mask = self.mask[idx]

        return row, masked_embedded_row, labels, mask