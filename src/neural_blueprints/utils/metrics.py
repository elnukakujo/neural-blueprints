import numpy as np
import torch

from .types import infer_types

def accuracy(y_pred: any, y_true: any, threshold: float = 0.05) -> float:
    """
    Calculates the accuracy metric.
    
    Args:
        y_pred (any): Predicted labels.
        y_true (any): True labels.
        threshold (float): Threshold to consider a prediction correct for regression tasks.

    Returns:
        float: Accuracy score.
    """
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()

    assert type(y_pred) == np.ndarray, f"y_pred must be a numpy array or convertible to one, but got {type(y_pred)}."
    assert type(y_true) == np.ndarray, f"y_true must be a numpy array or convertible to one, but got {type(y_true)}."

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    mask_dis = [v == 'int32' for v in infer_types(y_true).values()]

    result = np.empty_like(y_true, dtype=bool)
    for col_idx, is_discrete in enumerate(mask_dis):
        if is_discrete:
            # Discrete: exact match after rounding
            result[:, col_idx] = np.round(y_pred[:, col_idx]) == y_true[:, col_idx]
        else:
            # Continuous: within threshold
            result[:, col_idx] = np.abs(y_pred[:, col_idx] - y_true[:, col_idx]) <= threshold
    
    correct = np.sum(result)
    total = y_true.shape[0]
    return correct / total

def recall(y_pred: any, y_true: any, threshold: float = 0.05) -> float:
    """Calculates the recall metric.
    
    Args:
        y_pred (any): Predicted labels.
        y_true (any): True labels.
        threshold (float): Threshold to consider a prediction correct for regression tasks.
        
    Returns:
        float: Recall score.
    """
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()

    assert type(y_pred) == np.ndarray, f"y_pred must be a numpy array or convertible to one, but got {type(y_pred)}."
    assert type(y_true) == np.ndarray, f"y_true must be a numpy array or convertible to one, but got {type(y_true)}."

    mask_dis = [v == 'int32' for v in infer_types(y_true).values()]

    result = np.empty_like(y_true, dtype=bool)
    for col_idx, is_discrete in enumerate(mask_dis):
        if is_discrete:
            # Discrete: exact match after rounding
            result[:, col_idx] = np.round(y_pred[:, col_idx]) == y_true[:, col_idx]
        else:
            # Continuous: within threshold
            result[:, col_idx] = np.abs(y_pred[:, col_idx] - y_true[:, col_idx]) <= threshold

    # For recall: focus on positive samples in y_true
    positive_mask = y_true == 1

    true_positives = np.sum(result & positive_mask)
    false_negatives = np.sum(positive_mask)

    return true_positives / false_negatives

def f1_score(y_pred: any, y_true: any, threshold: float = 0.05) -> float:
    """Calculates the F1 score metric.
    
    Args:
        y_pred (any): Predicted labels.
        y_true (any): True labels.
        threshold (float): Threshold to consider a prediction correct for regression tasks.
        
    Returns:
        float: F1 score.
    """
    accuracy_score = accuracy(y_true, y_pred, threshold)
    recall_score = recall(y_true, y_pred, threshold)
    if (accuracy_score + recall_score) == 0:
        return 0.0
    
    return 2 * (accuracy_score * recall_score) / (accuracy_score + recall_score)