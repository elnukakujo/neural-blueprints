import torch
import torch.nn.functional as F

from .device import get_device

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
    elif metric_name == 'rmse':
        return rmse
    elif metric_name == 'mixed_type_reconstruction_loss':
        return mixed_type_reconstruction_loss
    elif metric_name == 'vae_loss':
        return vae_loss
    elif metric_name == 'nt_xent_loss':
        return nt_xent_loss
    elif metric_name == 'nt_bxent_loss':
        return nt_bxent_loss
    else:
        raise ValueError(f"Unsupported metric/loss function: {metric_name}")
    
def mse(y_pred, y_true) -> float:
    """Calculates the Mean Squared Error (MSE) metric.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        float: MSE score.
    """
    return F.mse_loss(y_pred, y_true)

def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculates the Root Mean Squared Error (RMSE) metric.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        float: RMSE score.
    """
    return F.mse_loss(y_pred, y_true, reduction='mean').sqrt()

def mae(y_pred, y_true) -> float:
    """Calculates the Mean Absolute Error (MAE) metric.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        
    Returns:
        float: MAE score.
    """
    return F.l1_loss(y_pred, y_true)

def cross_entropy(y_pred, y_true) -> float:
    """Calculates the Cross Entropy Loss metric.
    
    Args:
        y_pred (torch.Tensor): Predicted class probabilities (logits).
        y_true (torch.Tensor): True class labels (as integers).
    
    Returns:
        float: Cross Entropy Loss score.
    """
    return F.cross_entropy(y_pred, y_true)

def binary_cross_entropy(y_pred, y_true) -> float:
    """Calculates the Binary Cross Entropy Loss metric.
    
    Args:
        y_pred (torch.Tensor): Predicted probabilities for the positive class.
        y_true (torch.Tensor): True binary labels (0 or 1).
        
    Returns:
        float: Binary Cross Entropy Loss score.
    """
    return F.binary_cross_entropy(y_pred, y_true)

def mixed_type_reconstruction_loss(y_pred: list[torch.Tensor] | torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Computes the mixed-type reconstruction loss for tabular data, combining cross-entropy loss for categorical features and mean squared error (MSE) for continuous features.

    Args:
        y_pred (list[torch.Tensor]): List of predicted tensors for each feature. Each tensor has shape (batch_size, num_classes) for categorical features or (batch_size, 1) for continuous features.
        y_true (torch.Tensor): Ground truth tensor of shape (batch_size, num_features).

    Returns:
        torch.Tensor: Scalar tensor representing the mixed-type loss.
    """

    if isinstance(y_pred, list):
        total_loss = 0.0
        for col_idx in range(len(y_pred)):
            pred = y_pred[col_idx]
            label = y_true[:, col_idx]
            
            if pred.size(1) > 1:
                # Weighted cross-entropy
                loss = F.cross_entropy(input = pred, target = label.long(), reduction='sum')
            else:
                # MSE for continuous
                loss = F.mse_loss(pred.squeeze(-1), label.float(), reduction='sum')
            
            total_loss += loss
    elif isinstance(y_pred, torch.Tensor):
        if y_pred.shape == y_true.shape:
            # All continuous features
            total_loss = F.mse_loss(y_pred, y_true, reduction='sum')
    else:
        raise ValueError("y_pred must be either a list of tensors for a mixed-type scenario or a single tensor.")
    return total_loss / len(y_pred)

def vae_loss(y_pred: list[torch.Tensor], y_true: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Computes the Variational Autoencoder (VAE) loss, which combines reconstruction loss and KL divergence.

    Args:
        y_pred (list[torch.Tensor]): List of predicted tensors for each feature. Each tensor has shape (batch_size, num_classes) for categorical features or (batch_size, 1) for continuous features.
        y_true (torch.Tensor): Ground truth tensor of shape (batch_size, num_features).
        mu (torch.Tensor): Tensor of shape (batch_size, latent_dim) representing the mean of the latent distribution.
        logvar (torch.Tensor): Tensor of shape (batch_size, latent_dim) representing the log-variance of the latent distribution.
    
    Returns:
        torch.Tensor: Scalar tensor representing the VAE loss.
    """

    # Reconstruction loss
    recon_loss = mixed_type_reconstruction_loss(y_pred, y_true)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss + kl_loss

    return loss

def nt_xent_loss(proj: torch.Tensor, temperature: float = 1) -> torch.Tensor:
    """
    Computes the Normalized Temperature-scaled Cross Entropy (NT-Xent) loss for contrastive learning as described in the following Medium article:
    https://medium.com/data-science/nt-xent-normalized-temperature-scaled-cross-entropy-loss-explained-and-implemented-in-pytorch-cc081f69848
    
    Args:
        proj (torch.Tensor): Input tensor of shape (batch_size, projection_dim), representing the projection of the records.
    
    Returns:
        torch.Tensor: Scalar tensor representing the NT-Xent loss.
    """
    device = get_device()
    proj = proj.to(device)

    batch_size = proj.size(0)

    # Cosine similarity
    similarities = F.cosine_similarity(proj[None,:,:], proj[:,None,:], dim=-1).to(device)
    similarities[torch.eye(batch_size).bool()] = float("-inf")

    # Ground truth labels
    target = torch.arange(batch_size).to(device)
    target[0::2] += 1
    target[1::2] -= 1
    
    # Standard cross entropy loss
    return F.cross_entropy(similarities / temperature, target, reduction="mean")