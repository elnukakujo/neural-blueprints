import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable

from ..device import get_device


def get_eval_inference(evaluation_type: str) -> Callable:
    if 'vae' in evaluation_type:
        if any(loss in evaluation_type for loss in ['mse', 'mae', 'cross_entropy', 'binary_cross_entropy', 'root_mean_squared_error', 'reconstruction']):
            return eval_vae_base
    else:
        if 'nt_xent_loss' in evaluation_type:
            return eval_contrastive
        elif any(loss in evaluation_type for loss in ['mse', 'mae', 'cross_entropy', 'binary_cross_entropy', 'root_mean_squared_error', 'reconstruction']):
            return eval_base
    
    raise ValueError(f"Unknown evaluation type: {evaluation_type}")

def eval_contrastive(
        eval_loader: DataLoader,
        model: nn.Module,
        criterion: Callable,
    ) -> float:
    model.eval()
    total_loss = 0.0
    for batch in eval_loader:
        X_batch = batch[0].to(get_device())
        proj = model(X_batch)
        loss = criterion(proj = proj)
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss

def eval_vae_base(
        eval_loader: DataLoader,
        model: nn.Module,
        criterion: Callable
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in eval_loader:
        X_batch = batch[0].to(get_device())
        y_batch = batch[1].to(get_device())
        y_pred, mu, logvar = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = y_batch, mu=mu, logvar=logvar)
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss

def eval_base(
        eval_loader: DataLoader,
        model: nn.Module,
        criterion: Callable
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in eval_loader:
        X_batch = batch[0].to(get_device())
        y_batch = batch[1].to(get_device())
        y_pred = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = y_batch)
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss