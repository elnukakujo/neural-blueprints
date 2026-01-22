import torch
import torch.nn as nn
import pandas as pd
import os

from .base import BaseDataset
from ..architectures.base import EncoderArchitecture
from ..utils import get_device, infer_types

import logging
logger = logging.getLogger(__name__)

class TabularDataset(BaseDataset):
    def __init__(
            self, 
            data: pd.DataFrame,
            device: str = get_device()
        ):
        # Convert to tensor
        self.data = torch.tensor(data.to_numpy(), dtype=torch.float32, device=device)
        self.columns = data.columns.tolist()

        # Infer feature types
        data_types = infer_types(data)
        self.discrete_features = [col for col, dtype in data_types.items() if dtype == "int32"]
        self.continuous_features = [col for col, dtype in data_types.items() if dtype == "float32"]
        
        # Build cardinalities
        self.cardinalities = []
        
        for col in self.columns:
            if col in self.discrete_features:
                # Categorical: cardinality is number of unique values
                self.cardinalities.append(int(data[col].nunique()))
            elif col in self.continuous_features:
                # Continuous
                self.cardinalities.append(1)
            else:
                raise ValueError(f"Column {col} not found in discrete or continuous features.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "inputs": row
        }
    
class MaskedTabularDataset(TabularDataset):
    def __init__(
            self,
            data: pd.DataFrame,
            mask_prob: float = 0.1,
            mask_column: str | list[str] = None,
            device: str = get_device()
        ):
        TabularDataset.__init__(self, data, device=device)
        
        if mask_column is not None:
            # Mask entire column for all samples
            if isinstance(mask_column, str):
                mask_column = [mask_column]
            self.mask_column = mask_column
            col_idx = []
            for col in mask_column:
                col_idx.append(self.columns.index(col))
            self.mask = torch.zeros_like(self.data, dtype=torch.bool, device=device)
            self.mask[:, col_idx] = True
        else:
            # Randomly mask individual values across all samples and attributes
            gen = torch.Generator(device=device)
            seed = int(os.getenv("RANDOM_SEED")) if os.getenv("RANDOM_SEED") else torch.seed()
            gen.manual_seed(seed)
            self.mask = torch.rand(self.data.shape, device=device, generator=gen) < mask_prob
            if self.mask.all():
                logger.warning("All values are masked (mask is full of True). Consider lowering mask_prob.")
        
        self.labels = self.data.clone()
        
        nan_tokens = torch.zeros_like(self.data, device=device)
        for idx, cardinality in enumerate(self.cardinalities):
            if cardinality > 1:
                continue
            elif cardinality == 1:
                nan_tokens[:, idx] = -1  # For continuous features
            else:
                raise ValueError(f"Cardinality {cardinality} for column {self.columns[idx]} is invalid.")
        
        # Apply masking using torch.where
        # Where mask is True: use nan_token, otherwise keep original value
        self.data = torch.where(self.mask, nan_tokens, self.data)

        # For labels: where mask is False (not masked), use nan_token
        self.labels = torch.where(~self.mask, nan_tokens, self.labels)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        labels = self.labels[idx]
        mask = self.mask[idx]
        
        return {
            "inputs": row,
            "label": labels,
            "metadata": {"mask": mask}
        }
    
class FeatureTabularDataset(TabularDataset):
    def __init__(
            self,
            column_encoders: dict[str, EncoderArchitecture],
            data: pd.DataFrame,
            device: str = get_device()
        ):
        columns = data.columns.tolist()
        self.representations = {}

        columns_encoded = []
        for column_to_encode, representation_model in column_encoders.items():
            cols = column_to_encode.split(",") if "," in column_to_encode else [column_to_encode]
            columns_encoded.extend(cols)
            key = column_to_encode

            # Get data for specified columns
            col_indices = [columns.index(col) for col in cols if col in columns]
            data_to_encode = data.iloc[:, col_indices].to_numpy()
            
            data_to_encode = torch.tensor(data_to_encode, device=device)
            representation_model = representation_model.to(device)

            # Generate representations from the specified attributes
            with torch.no_grad():
                self.representations[key] = representation_model.encode(data_to_encode)

        if len(column_encoders.keys()) == 1:
            # If only one representation, store as single tensor
            self.representations: torch.Tensor = list(self.representations.values())[0]

        # Handle classic tabular data (not encoded)
        columns_not_encoded = [col for col in columns if col not in columns_encoded]
        if len(columns_not_encoded) == 0:
            data = pd.DataFrame()
        else:
            data = data.iloc[:, [columns.index(col) for col in columns_not_encoded]]
            TabularDataset.__init__(self, data, device=device)
    
    def __getitem__(self, idx):
        tabular_data = self.data[idx]
        if isinstance(self.representations, dict):
            representation_data = {key: value[idx] for key, value in self.representations.items()}
        else:
            representation_data = self.representations[idx]

        return {
            "inputs": {
                "tabular": tabular_data,
                "representation": representation_data
            }
        }
    
class TabularLabelDataset(TabularDataset):
    def __init__(
            self,
            data: pd.DataFrame,
            label_columns: list[str],
            device: str = get_device()
        ):
        # Get indices of label columns, extract label tensors and store in dictionary
        if len(label_columns) == 1:
            labels_data = data[label_columns[0]].to_numpy()
            self.labels = torch.tensor(labels_data, dtype=torch.float32, device=device)
        else:
            self.labels = {
                col: torch.tensor(data[col].to_numpy(), dtype=torch.float32, device=device) 
                for col in label_columns
            }

        data = data.drop(columns=label_columns)

        TabularDataset.__init__(self, data, device=device)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        if isinstance(self.labels, dict):
            label = {k: v[idx] for k, v in self.labels.items()}
        else:
            label = self.labels[idx]
        return {
            "inputs": row,
            "label": label
        }

class FeatureTabularLabelDataset(FeatureTabularDataset, TabularLabelDataset):
    def __init__(
            self,
            column_encoders: dict[str, EncoderArchitecture],
            data: pd.DataFrame,
            label_column: list[str],
            device: str = get_device()
        ):
        data_without_labels = data.drop(columns=label_column)

        TabularLabelDataset.__init__(self, data, label_column, device=device)
        FeatureTabularDataset.__init__(self, column_encoders, data_without_labels, device=device)

    def __getitem__(self, idx):
        tabular_data = self.data[idx]

        if isinstance(self.representations, dict):
            representation_data = {key: value[idx] for key, value in self.representations.items()}
        else:
            representation_data = self.representations[idx]
        
        if isinstance(self.labels, dict):
            label = {k: v[idx] for k, v in self.labels.items()} 
        else:
            label = self.labels[idx]

        return {
            "inputs": {
                "tabular": tabular_data,
                "representation": representation_data
            },
            "label": label
        }