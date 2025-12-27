import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable

from ..device import get_device

def get_eval_inference(evaluation_type: str) -> Callable:
    if 'vae' in evaluation_type:
        if  'reconstruction' in evaluation_type:
            return eval_vae_reconstruction
        elif 'masked_reconstruction' in evaluation_type:
            return eval_vae_masked_reconstruction
        elif any(loss in evaluation_type for loss in ['mse', 'mae', 'cross_entropy', 'binary_cross_entropy', 'root_mean_squared_error']):
            return eval_vae_base
        else:
            raise ValueError(f"Unknown VAE training type: {evaluation_type}")
    else:
        if 'nt_xent_loss' in evaluation_type:
            return eval_contrastive
        elif 'reconstruction' in evaluation_type:
            return eval_reconstruction
        elif 'masked_reconstruction' in evaluation_type:
            return eval_masked_reconstruction
        elif any(loss in evaluation_type for loss in ['mse', 'mae', 'cross_entropy', 'binary_cross_entropy', 'root_mean_squared_error']):
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

def eval_vae_masked_reconstruction(
        eval_loader: DataLoader,
        model: nn.Module,
        criterion: Callable
) -> float:
    model.train()
    total_loss = 0.0
    for batch in eval_loader:
        X_batch = batch[0].to(get_device())
        y_batch = batch[1].to(get_device())
        mask = batch[2].to(get_device())
        y_pred, mu, logvar = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = y_batch, mu=mu, logvar=logvar, mask=mask)
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss

def eval_masked_reconstruction(
        eval_loader: DataLoader,
        model: nn.Module,
        criterion: Callable
) -> float:
    model.train()
    total_loss = 0.0
    for batch in eval_loader:
        X_batch = batch[0].to(get_device())
        y_batch = batch[1].to(get_device())
        mask = batch[2].to(get_device())
        y_pred = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = y_batch, mask=mask)
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss

def eval_vae_reconstruction(
        eval_loader: DataLoader,
        model: nn.Module,
        criterion: Callable
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in eval_loader:
        if isinstance(batch, list) or isinstance(batch, tuple):
            X_batch = batch[0].to(get_device())
        else:
            X_batch = batch.to(get_device())
        y_pred, mu, logvar = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = X_batch, mu=mu, logvar=logvar)
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(eval_loader.dataset)
    return avg_loss

def eval_reconstruction(
        eval_loader: DataLoader,
        model: nn.Module,
        criterion: Callable
) -> float:
    model.eval()
    total_loss = 0.0
    for batch in eval_loader:
        X_batch = batch[0].to(get_device())
        y_pred = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = X_batch)
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