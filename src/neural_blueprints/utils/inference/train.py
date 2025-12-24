import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable

from ..device import get_device

import logging
logger = logging.getLogger(__name__)

def get_train_inference(training_type: str) -> Callable:
    if 'vae' in training_type:
        if any(loss in training_type for loss in ['mse', 'mae', 'cross_entropy', 'binary_cross_entropy', 'root_mean_squared_error', 'reconstruction']):
            return train_vae_base
    else:
        if 'nt_xent_loss' in training_type:
            return train_contrastive
        elif any(loss in training_type for loss in ['mse', 'mae', 'cross_entropy', 'binary_cross_entropy', 'root_mean_squared_error', 'reconstruction']):
            return train_base
    
    raise ValueError(f"Unknown training type: {training_type}")

def train_contrastive(
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
    ) -> float:
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        X_batch = batch[0].to(get_device())
        optimizer.zero_grad()
        proj = model(X_batch)
        loss = criterion(proj = proj)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def train_vae_base(
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable
) -> float:
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        X_batch = batch[0].to(get_device())
        y_batch = batch[1].to(get_device())
        optimizer.zero_grad()
        y_pred, mu, logvar = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = y_batch, mu=mu, logvar=logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def train_base(
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable
) -> float:
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        X_batch = batch[0].to(get_device())
        y_batch = batch[1].to(get_device())
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred = y_pred, y_true = y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss