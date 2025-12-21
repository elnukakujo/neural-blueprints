import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable

from ..device import get_device

def get_eval_epoch(evaluation_type: str) -> Callable:
    if evaluation_type in ['nt_xent_loss']:
        return eval_epoch_contrastive
    elif evaluation_type in ['vae_loss']:
        return eval_epoch_vae
    elif evaluation_type in ['mse', 'mae', 'cross_entropy', 'binary_cross_entropy', 'root_mean_squared_error', 'mixed_type_reconstruction_loss']:
        return eval_epoch_base
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")

def eval_epoch_contrastive(
        eval_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
    ) -> float:
    model.eval()
    total_loss = 0.0
    for X_batch, _ in eval_loader:
        X_batch = X_batch.to(get_device())
        optimizer.zero_grad()
        proj = model(X_batch)
        loss = criterion(proj = proj)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss

def eval_epoch_vae(
        eval_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable
) -> float:
    model.eval()
    total_loss = 0.0
    for X_batch, _ in eval_loader:
        X_batch = X_batch.to(get_device())
        optimizer.zero_grad()
        X_reconstructed, mu, logvar = model(X_batch)
        loss = criterion(y_pred = X_reconstructed, y_true = X_batch, mu=mu, logvar=logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss

def eval_epoch_base(
        eval_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in eval_loader:
        X_batch, y_batch = batch[0].to(get_device()), batch[1].to(get_device())
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss