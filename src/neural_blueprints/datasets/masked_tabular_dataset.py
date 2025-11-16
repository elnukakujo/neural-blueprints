import torch
from torch.utils.data import Dataset

class MaskedTabularDataset(Dataset):
    def __init__(self, data, discrete_features, continuous_features, mask_prob=0.15):
        # Convert to tensor
        self.data = torch.tensor(data.to_numpy(), dtype=torch.float32)
        self.columns = data.columns.tolist()
        
        # Build cardinalities and token indices
        self.cardinalities = []
        self.mask_tokens = []
        self.nan_tokens = []
        
        for col in self.columns:
            if col in discrete_features:
                # Categorical: cardinality is number of unique values
                cardinality = int(data[col].max() + 1)
                self.cardinalities.append(cardinality)
                self.mask_tokens.append(float(1))  # Always 1
                self.nan_tokens.append(float(0))    # Always 0
            else:
                # Continuous
                self.cardinalities.append(1)
                self.mask_tokens.append(2.0)      # Use 0.0 (standardized mean)
                self.nan_tokens.append(-2.0)    # Special ignore value
        
        self.dis_idx = [self.columns.index(c) for c in discrete_features]
        self.cont_idx = [self.columns.index(c) for c in continuous_features]
        self.mask_prob = mask_prob
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx].clone()
        labels = row.clone()
        
        # Create mask
        mask = torch.rand(len(row)) < self.mask_prob
        
        # Apply mask tokens
        for col_idx in range(len(row)):
            if mask[col_idx]:
                # Mask this position
                row[col_idx] = self.mask_tokens[col_idx]
                # Label keeps original value
            else:
                # Don't mask
                # Row keeps original value
                labels[col_idx] = self.nan_tokens[col_idx]
        
        return row, labels, mask