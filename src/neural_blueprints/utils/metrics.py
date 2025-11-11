"""Evaluation metrics"""
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        
    Returns:
        Accuracy as percentage
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    
    return 100.0 * correct / total


def compute_f1(predictions: torch.Tensor, targets: torch.Tensor, 
               average: str = 'weighted') -> float:
    """
    Compute F1 score
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        average: Averaging method ('micro', 'macro', 'weighted')
        
    Returns:
        F1 score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    return f1_score(targets_np, preds_np, average=average, zero_division=0)


def compute_precision(predictions: torch.Tensor, targets: torch.Tensor,
                     average: str = 'weighted') -> float:
    """Compute precision score"""
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    return precision_score(targets_np, preds_np, average=average, zero_division=0)


def compute_recall(predictions: torch.Tensor, targets: torch.Tensor,
                  average: str = 'weighted') -> float:
    """Compute recall score"""
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    return recall_score(targets_np, preds_np, average=average, zero_division=0)


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity
    """
    return np.exp(loss)