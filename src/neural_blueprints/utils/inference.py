import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Optional
from dataclasses import dataclass

from .device import get_device

import logging
logger = logging.getLogger(__name__)

def get_inference(training_type: str) -> tuple[Callable, Callable]:
    """Factory function that returns configured training function"""
    
    # Determine batch configuration
    if 'masked_reconstruction' in training_type:
        batch_config = BatchConfig(
            input_idx=0,
            target_idx=1,
            mask_idx=2
        )
    elif 'reconstruction' in training_type:
        batch_config = BatchConfig(use_input_as_target=True)
    elif 'nt_xent_loss' in training_type:
        batch_config = BatchConfig(target_idx=None)
    else:
        batch_config = BatchConfig()  # Default: supervised learning
    
    # Determine model configuration
    model_config = ModelConfig(
        is_vae='vae' in training_type,
        is_contrastive='nt_xent_loss' in training_type
    )
    
    # Return partially applied function
    def train_fn(train_loader, model, optimizer, criterion):
        return train_unified(
            train_loader, model, optimizer, criterion,
            batch_config, model_config
        )
    
    def eval_fn(eval_loader, model, criterion):
        return eval_unified(
            eval_loader, model, criterion,
            batch_config, model_config
        )
    
    return train_fn, eval_fn

@dataclass
class BatchConfig:
    """Configuration for how to extract data from batches"""
    input_idx: int = 0
    target_idx: Optional[int] = 1
    mask_idx: Optional[int] = None
    use_input_as_target: bool = False  # For reconstruction tasks
    is_nt_xent: bool = False  # For contrastive learning

@dataclass
class ModelConfig:
    """Configuration for model output handling"""
    is_vae: bool = False  # Returns (output, mu, logvar)
    is_contrastive: bool = False  # For contrastive learning


def get_batch_data(batch, config: BatchConfig, device):
    """Extract inputs, targets, and masks from batch based on config"""
    # Handle single tensor vs tuple/list
    if not isinstance(batch, (list, tuple)):
        X = batch.to(device)
        y = X if config.use_input_as_target else None
        mask = None
    else:
        X = batch[config.input_idx].to(device)

        if config.is_nt_xent:
            y = None  # No targets for contrastive learning

            combined = torch.empty((X.size(0) * 2, X.size(1)), device=device)
            X_full = X.clone()
            X_full[batch[config.mask_idx]] = batch[config.target_idx]
            combined[0::2] = X_full
            combined[1::2] = X
            
            X = combined

        
        if config.use_input_as_target:
            y = X
        elif config.target_idx is not None:
            y = batch[config.target_idx].to(device)
        else:
            y = None
            
        if config.mask_idx is not None:
            mask = batch[config.mask_idx].to(device)
        else:
            mask = None
    
    return X, y, mask


def compute_loss(model_output:tuple | torch.Tensor, y_true: torch.Tensor, criterion: Callable, mask, model_config: ModelConfig):
    """Compute loss handling different model output types"""
    # Unpack model output
    if model_config.is_vae:  # VAE: (y_pred, mu, logvar)
        assert isinstance(model_output, tuple) and len(model_output) == 3, "VAE model output must be a tuple of (y_pred, mu, logvar)"

        y_pred, mu, logvar = model_output
        loss_kwargs = {'y_pred': y_pred, 'y_true': y_true, 
                        'mu': mu, 'logvar': logvar}
    elif model_config.is_contrastive:  # Contrastive learning
        y_pred = model_output  # Typically projections
        loss_kwargs = {'proj': y_pred}
    else:
        y_pred = model_output
        loss_kwargs = {'y_pred': y_pred, 'y_true': y_true}
    
    # Add mask if provided
    if mask is not None:
        loss_kwargs['mask'] = mask
    
    return criterion(**loss_kwargs)


def train_unified(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    batch_config: BatchConfig,
    model_config: ModelConfig
) -> float:
    """Unified training function for all model types"""
    model.train()
    total_loss = 0.0
    device = get_device()
    
    for batch in train_loader:
        # Extract data from batch
        X, y, mask = get_batch_data(batch, batch_config, device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(X)
        
        # Compute loss
        loss = compute_loss(output, y, criterion, mask, model_config)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def eval_unified(
    eval_loader: DataLoader,
    model: nn.Module,
    criterion: Callable,
    batch_config: BatchConfig,
    model_config: ModelConfig
) -> float:
    """Unified evaluation function for all model types"""
    model.eval()
    total_loss = 0.0
    device = get_device()
    
    with torch.no_grad():
        for batch in eval_loader:
            # Extract data from batch
            X, y, mask = get_batch_data(batch, batch_config, device)
            
            # Forward pass
            output = model(X)
            
            # Compute loss
            loss = compute_loss(output, y, criterion, mask, model_config)
            
            total_loss += loss.item() * X.size(0)
    
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss