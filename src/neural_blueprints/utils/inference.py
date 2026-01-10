import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from dataclasses import dataclass

from .device import get_device

import logging
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model output handling"""
    is_vae: bool = False          # Returns (output, mu, logvar)
    is_contrastive: bool = False  # For contrastive learning
    is_unsupervised: bool = False  # No labels provided


def compute_loss(model_output, inputs: dict, y_true, criterion: Callable, mask, model_config: ModelConfig):
    """Compute loss handling different model output types"""
    if model_config.is_vae:  # VAE: (y_pred, mu, logvar)
        assert isinstance(model_output, tuple) and len(model_output) == 3, \
            "VAE model output must be a tuple of (y_pred, mu, logvar)"
        y_pred, mu, logvar = model_output
        loss_kwargs = {'y_pred': y_pred, 'y_true': y_true, 'mu': mu, 'logvar': logvar}
    elif model_config.is_contrastive:
        y_pred = model_output
        loss_kwargs = {'proj': y_pred}
    else:
        y_pred = model_output
        if model_config.is_unsupervised:
            y_true = inputs  # Use inputs as targets for unsupervised
        loss_kwargs = {'y_pred': y_pred, 'y_true': y_true}

    if mask is not None:
        loss_kwargs['mask'] = mask

    return criterion(**loss_kwargs)


def train_unified(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    model_config: ModelConfig = ModelConfig()
) -> float:
    """Unified training function for both UniModal and MultiModal samples"""
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        # Batch is always a dict with keys: 'inputs', 'label', 'metadata'
        X = batch["inputs"]
        y = batch.get("label", None)
        mask = batch.get("metadata", {}).get("mask") if batch.get("metadata") else None

        optimizer.zero_grad()
        output = model(X)
        loss = compute_loss(output, X, y, criterion, mask, model_config)
        loss.backward()
        optimizer.step()

        # Use batch size for averaging
        if isinstance(X, dict):
            batch_size = next(iter(X.values())).size(0)  # first modality
        else:
            batch_size = X.size(0)
        total_loss += loss.item() * batch_size

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def eval_unified(
    eval_loader: DataLoader,
    model: nn.Module,
    criterion: Callable,
    model_config: ModelConfig = ModelConfig()
) -> float:
    """Unified evaluation function for both UniModal and MultiModal samples"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in eval_loader:
            X = batch["inputs"]
            y = batch.get("label", None)
            mask = batch.get("metadata", {}).get("mask") if batch.get("metadata") else None

            output = model(X)
            loss = compute_loss(output, X, y, criterion, mask, model_config)

            if isinstance(X, dict):
                batch_size = next(iter(X.values())).size(0)
            else:
                batch_size = X.size(0)
            total_loss += loss.item() * batch_size

    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss


def get_inference(training_type: str) -> tuple[Callable, Callable]:
    """
    Factory function that returns configured training and evaluation functions.
    ModelConfig flags are inferred from training_type.
    """
    model_config = ModelConfig(
        is_vae='vae' in training_type,
        is_contrastive='nt_xent_loss' in training_type,
        is_unsupervised=training_type in ['reconstruction', 'vae_reconstruction']
    )

    def train_fn(train_loader, model, optimizer, criterion):
        return train_unified(train_loader, model, optimizer, criterion, model_config)

    def eval_fn(eval_loader, model, criterion):
        return eval_unified(eval_loader, model, criterion, model_config)

    return train_fn, eval_fn