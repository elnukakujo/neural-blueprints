import numpy as np
import torch
import torch.nn as nn

def get_criterion(metric_name: str) -> callable:
    """Returns the appropriate loss function based on the metric name.
    
    Args:
        metric_name (str): Name of the metric/loss function.
        
    Returns:
        Callable: Corresponding loss function.
    """
    metric_name = metric_name.lower()
    if metric_name == 'mse':
        return mse
    elif metric_name == 'mae':
        return mae
    elif metric_name == 'cross_entropy':
        return cross_entropy
    elif metric_name == 'binary_cross_entropy':
        return binary_cross_entropy
    elif metric_name == 'vae_loss':
        return vae_loss
    elif metric_name == 'tabular_masked_loss':
        return tabular_masked_loss
    else:
        raise ValueError(f"Unsupported metric/loss function: {metric_name}")

def get_optimizer(optimizer_name: str, parameters, lr: float = 1e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Returns the appropriate optimizer based on the optimizer name.
    
    Args:
        optimizer_name (str): Name of the optimizer.
        parameters: Model parameters to optimize.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization).
        
    Returns:
        torch.optim.Optimizer: Corresponding optimizer.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.05, mask_dis: torch.BoolTensor = None) -> float:
    """Calculates the accuracy metric.
    
    Args:
        y_pred (torch.Tensor): Predicted labels.
        y_true (torch.Tensor): True labels.
        threshold (float): Threshold to consider a prediction correct for regression tasks.
        mask_dis (torch.BoolTensor, optional): Boolean mask indicating which samples are categorical.

    Returns:
        float: Accuracy score.
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    if mask_dis is None:
        mask_dis = np.isclose(y_true_np, np.round(y_true_np), atol=1e-8)
        if mask_dis.ndim > 1:
            mask_dis = np.all(mask_dis, axis=1)
        else:
            mask_dis = mask_dis.astype(bool)

    result = np.empty_like(y_true_np, dtype=bool)
    result[mask_dis] = np.round(y_pred_np[mask_dis]) == y_true_np[mask_dis]
    result[~mask_dis] = np.abs(y_pred_np[~mask_dis] - y_true_np[~mask_dis]) <= threshold
    
    correct = np.sum(result)
    total = y_true_np.shape[0]
    return correct / total

def recall(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.05, mask_dis: torch.BoolTensor = None) -> float:
    """Calculates the recall metric.
    
    Args:
        y_pred (torch.Tensor): Predicted labels.
        y_true (torch.Tensor): True labels.
        threshold (float): Threshold to consider a prediction correct for regression tasks.
        mask_dis (torch.BoolTensor, optional): Boolean mask indicating which samples are categorical.
        
    Returns:
        float: Recall score.
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    if mask_dis is None:
        mask_dis = np.isclose(y_true_np, np.round(y_true_np), atol=1e-8)
        mask_dis = np.all(mask_dis, axis=1)

    # Initialize boolean result array
    correct_mask = np.zeros_like(y_true_np, dtype=bool)

    # Discrete rows: exact match
    correct_mask[mask_dis] = np.round(y_pred_np[mask_dis]) == y_true_np[mask_dis]

    # Continuous rows: within relative threshold
    correct_mask[~mask_dis] = (
        np.abs(y_pred_np[~mask_dis] - y_true_np[~mask_dis])
        / (np.abs(y_true_np[~mask_dis]) + 1e-12)
    ) <= threshold

    # For recall: focus on positive samples in y_true
    positive_mask = y_true_np == 1
    if np.sum(positive_mask) == 0:
        return 1.0  # no positives, perfect recall by definition

    true_positives = np.sum(correct_mask & positive_mask)
    false_negatives = np.sum(positive_mask)

    return true_positives / false_negatives

def f1_score(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.05, mask_dis: torch.BoolTensor = None) -> float:
    """Calculates the F1 score metric.
    
    Args:
        y_pred (torch.Tensor): Predicted labels.
        y_true (torch.Tensor): True labels.
        threshold (float): Threshold to consider a prediction correct for regression tasks.
        mask_dis (torch.BoolTensor, optional): Boolean mask indicating which samples are categorical.
        
    Returns:
        float: F1 score.
    """
    accuracy_score = accuracy(y_true, y_pred, threshold, mask_dis)
    recall_score = recall(y_true, y_pred, threshold, mask_dis)
    if (accuracy_score + recall_score) == 0:
        return 0.0
    
    return 2 * (accuracy_score * recall_score) / (accuracy_score + recall_score)

def mse(y_pred, y_true) -> float:
    """Calculates the Mean Squared Error (MSE) metric.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        float: MSE score.
    """
    return torch.nn.functional.mse_loss(y_pred, y_true)

def mae(y_pred, y_true) -> float:
    """Calculates the Mean Absolute Error (MAE) metric.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        float: MAE score.
    """
    return torch.nn.functional.l1_loss(y_pred, y_true)

def cross_entropy(y_pred, y_true) -> float:
    """Calculates the Cross Entropy Loss metric.
    
    Args:
        y_pred (torch.Tensor): Predicted class probabilities (logits).
        y_true (torch.Tensor): True class labels (as integers).
    
    Returns:
        float: Cross Entropy Loss score.
    """
    return nn.functional.cross_entropy(y_pred, y_true)

def binary_cross_entropy(y_pred, y_true) -> float:
    """Calculates the Binary Cross Entropy Loss metric.
    
    Args:
        y_pred (torch.Tensor): Predicted probabilities for the positive class.
        y_true (torch.Tensor): True binary labels (0 or 1).
        
    Returns:
        float: Binary Cross Entropy Loss score.
    """
    return nn.functional.binary_cross_entropy(y_pred, y_true)

def vae_loss(y_pred, y_true):
    """Calculates the Variational AutoEncoder (VAE) loss.

    Args:
        y_pred (tuple): Tuple containing reconstructed data, mean, and log variance from the VAE.
        y_true (torch.Tensor): Original input data.
    
    Returns:
        torch.Tensor: VAE loss value.
    """

    # Reconstruction loss (MSE)
    recon_x, mu, logvar = y_pred
    x = y_true

    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def root_mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates the Root Mean Squared Error (RMSE) metric.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        float: RMSE score.
    """
    return nn.functional.mse_loss(y_pred, y_true, reduction='mean').sqrt()

def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates the R-squared (R2) metric.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        float: R2 score.
    """
    return nn.functional.r2_score(y_pred, y_true)

def tabular_masked_loss(y_pred, y_true, mask):
    """Calculates the masked loss for tabular data with mixed feature types (discrete and continuous).
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        mask (torch.BoolTensor): Boolean mask indicating which entries to include in the loss.
        
    Returns:
        float: Computed masked loss value.
    """
    total_loss = 0
        
    for col_idx in range(len(y_pred)):
        pred = y_pred[col_idx]
        label = y_true[:, col_idx]
        col_mask = mask[:, col_idx]
        
        if col_mask.sum() == 0:
            continue
        
        if pred.size(1) > 1:
            # Weighted cross-entropy
            loss = nn.functional.cross_entropy(pred[col_mask], label[col_mask].long(), reduction='mean')
        else:
            # MSE for continuous
            loss = nn.functional.mse_loss(pred[col_mask].squeeze(-1), label[col_mask].float(), reduction='mean')
        
        total_loss += loss
    
    return total_loss / len(y_pred)