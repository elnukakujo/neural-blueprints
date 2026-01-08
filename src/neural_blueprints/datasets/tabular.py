import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset

from .schemas import UniModalSample, MultiModalSample
from ..architectures.base import EncoderArchitecture
from ..utils import get_device

import logging
logger = logging.getLogger(__name__)

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
                logger.warning(f"Column {col} not found in discrete or continuous features.")
        self.nan_tokens = torch.tensor(self.nan_tokens, dtype=self.data.dtype, device=device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        return UniModalSample(data=row, label=None)
    
class TabularLabelDataset(TabularDataset):
    """
    PyTorch Dataset for tabular data with  label classification/regression.

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
            label_columns: list[str],
            device: str = None
        ):
        super().__init__(data, discrete_features, continuous_features, device=device)

        # Get indices of label columns, extract label tensors and store in dictionary
        if len(label_columns) == 1:
            label_idx = self.columns.index(label_columns[0])
            self.labels = self.data[:, label_idx]
        else:
            label_idx = [self.columns.index(col) for col in label_columns]
            self.labels = {col: self.data[:, i] for col, i in zip(label_columns, label_idx)}

        # Remove label columns from data tensor
        keep_idx = [i for i in range(self.data.shape[1]) if i not in label_idx]
        self.data = self.data[:, keep_idx]

        # Remove corresponding cardinalities and nan_tokens
        self.cardinalities = [self.cardinalities[i] for i in keep_idx]
        self.nan_tokens = self.nan_tokens[keep_idx]

        # Update columns list
        self.columns = [self.columns[i] for i in keep_idx]
    
    def __getitem__(self, idx):
        row = self.data[idx]
        if isinstance(self.labels, dict):
            label = {k: v[idx] for k, v in self.labels.items()}
        else:
            label = self.labels[idx]
        return UniModalSample(data=row, label=label)
    
class FeatureTabularLabelDataset(TabularLabelDataset):
    def __init__(
            self,
            column_encoders: dict[str, EncoderArchitecture],
            data: pd.DataFrame,
            discrete_features: list[str],
            continuous_features: list[str],
            label_column: str,
            device: str = None
        ):
        super().__init__(data, discrete_features, continuous_features, label_column, device=device)

        self.representations = {}

        for column_to_encode, representation_model in column_encoders.items():
            cols = column_to_encode.split(",") if "," in column_to_encode else [column_to_encode]
            key = column_to_encode

            # Get data for specified columns
            col_indices = [self.columns.index(col) for col in cols if col in self.columns]
            data_to_encode = self.data[:, col_indices]

            device = get_device() if device is None else device
            representation_model = representation_model.to(device)
            data_to_encode = data_to_encode.to(device)

            # Generate representations from the specified attributes
            with torch.no_grad():
                self.representations[key] = representation_model.encode(data_to_encode)

        # Handle classic tabular data (not encoded)
        columns_not_encoded = [col for col in self.columns if col not in column_encoders]
        self.tabular = self.data[:, [self.columns.index(col) for col in columns_not_encoded]]

    def __getitem__(self, idx):
        tabular_data = self.tabular[idx]
        representation_data = {key: value[idx] for key, value in self.representations.items()}
        if isinstance(self.labels, dict):
            label = {k: v[idx] for k, v in self.labels.items()} 
        else:
            label = self.labels[idx]

        return MultiModalSample(
            tabular=tabular_data,
            representation=representation_data,
            label=label
        )

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
        
        return UniModalSample(
            data=row,
            label=labels,
            metadata={"mask": mask}
        )
    
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