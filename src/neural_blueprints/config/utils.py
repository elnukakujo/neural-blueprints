from typing import Optional
from pydantic import BaseModel, model_validator

class TrainerConfig(BaseModel):
    """Configuration schema for training hyperparameters and settings.
    
    Args:
        learning_rate (float): Initial learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) factor.
        batch_size (int): Number of samples per training batch.
        save_weights_path (Optional[str]): Path to save model weights. If None, weights are not saved.
        training_type (str): Type of training to perform. Options are 'reconstruction', 'masked_label', 'label'.
        criterion (str): Loss function to use. Options depend on training_type.
        optimizer (str): Optimizer to use for training. Options include 'adam', 'sgd', etc.
    """
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    save_weights_path: Optional[str] = None
    training_type: str = "label"  # Options: 'reconstruction', 'masked_label', 'label'
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
        valid_types = ['reconstruction', 'masked_label', 'label']
        if self.training_type not in valid_types:
            raise ValueError(f"Invalid training_type: {self.training_type}. Must be one of {valid_types}.")
        if self.training_type == 'reconstruction' and self.criterion not in ['mse', 'mae', 'vae_loss']:
            raise ValueError("For 'reconstruction' training_type, criterion must be one of ['mse', 'mae', 'vae_loss'].")
        if self.training_type == 'masked_label' and self.criterion not in ['tabular_masked_loss']:
            raise ValueError("For 'masked_label' training_type, criterion must be 'tabular_masked_loss'.")
        if self.training_type == 'label' and self.criterion not in ['cross_entropy', 'binary_cross_entropy', 'mse', 'mae', 'rmse']:
            raise ValueError("For 'label' training_type, criterion must be one of ['cross_entropy', 'binary_cross_entropy', 'mse', 'mae', 'rmse'].")
        return self