from typing import Optional
from pydantic import BaseModel, model_validator

class TrainerConfig(BaseModel):
    """Configuration schema for training hyperparameters and settings.
    
    Args:
        learning_rate (float): Initial learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) factor.
        batch_size (int): Number of samples per training batch.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
        save_weights_path (Optional[str]): Path to save model weights. If None, weights are not saved.
        criterion (str): Loss function to use. Options include 'mse', 'mae', 'cross_entropy', etc.
        optimizer (str): Optimizer to use for training. Options include 'adam', 'sgd', etc.
    """
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    early_stopping_patience: int = 10
    save_weights_path: Optional[str] = None
    criterion: str = "mse"  # Options: 'mse', 'mae', 'cross_entropy', etc.
    optimizer: str = "adam"  # Options: 'adam', 'sgd', etc.

    @model_validator(mode="after")
    def _validate(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.early_stopping_patience < 0:
            raise ValueError("Early stopping patience cannot be negative.")
        return self