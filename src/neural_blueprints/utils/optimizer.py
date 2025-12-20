
import torch

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