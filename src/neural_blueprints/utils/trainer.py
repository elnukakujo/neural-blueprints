import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time
import logging
logger = logging.getLogger(__name__)

from .criterion import get_criterion
from .optimizer import get_optimizer
from .device import get_device
from .visualize import curve_plot
from .inference import get_train_inference, get_eval_inference, get_predict_inference
from ..config.utils import TrainerConfig

PACKAGE_DIR = os.path.dirname(__file__)  # points to github_repo/src/neural_blueprints/utils
PARENT_DIR = os.path.dirname(PACKAGE_DIR)  # points to github_repo/src/neural_blueprints
TEMP_DIR = os.path.join(PARENT_DIR, "temp")

class Trainer:
    def __init__(
            self,
            config: TrainerConfig,
            model: nn.Module
        ):
        self.device = get_device()
        self.model = model.to(self.device)

        self.criterion = get_criterion(config.criterion)
        self.train_epoch = get_train_inference(config.criterion)
        self.eval_epoch = get_eval_inference(config.criterion)
        self.predict_inference = get_predict_inference(config.criterion)
        self.optimizer = get_optimizer(config.optimizer, model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.early_stopping_patience = config.early_stopping_patience

        self.save_weights_path = config.save_weights_path
        if self.save_weights_path is not None:
            if not os.path.exists(os.path.dirname(self.save_weights_path)):
                os.makedirs(os.path.dirname(self.save_weights_path), exist_ok=False)
            else:
                if os.path.isfile(self.save_weights_path):
                    os.remove(self.save_weights_path)
                print(f"Directory {os.path.dirname(self.save_weights_path)} already exists. Existing weights are overwritten.")

        self.best_val_loss = float('inf')
        self.epoch_since_improvement = 0
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        self.batch_size = config.batch_size
        self.model_path = os.path.join(TEMP_DIR, "best_model.pth")

        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train(self, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, epochs: int, visualize: bool = True):
        train_losses = []
        val_losses = []

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        start_time = time.time()

        for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
            train_losses.append(self.train_epoch(
                train_loader,
                self.model,
                self.optimizer,
                self.criterion
            ))
            val_losses.append(self.eval_epoch(
                val_loader,
                self.model,
                self.criterion
            ))
            if self.best_val_loss == float('inf'):
                self.best_val_loss = val_losses[-1]
                torch.save(self.model.state_dict(), self.model_path)
            elif self.best_val_loss - val_losses[-1] >= 1e-3:
                self.best_val_loss = val_losses[-1]
                self.epoch_since_improvement = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                self.epoch_since_improvement += 1

            if self.epoch_since_improvement >= self.early_stopping_patience:
                logger.info(f"No improvement in validation loss for {self.epoch_since_improvement} consecutive epochs. Early stopping at epoch {epoch+1}.")
                break

            tqdm.write(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

        end_time = time.time()
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")
        logger.info(f"Best validation loss: {self.best_val_loss:.4e}")

        if visualize:
            curve_plot(
                x=list(range(1, epochs + 1)),
                ys=[train_losses, val_losses],
                labels=["Training Loss", "Validation Loss"],
                title="Training and Validation Loss over Epochs",
                x_label="Epochs",
                y_label="Loss",
                show_min_max=True
            )

        if self.best_val_loss < float('inf') and self.best_val_loss != float('nan'):
            self.model.load_state_dict(torch.load(self.model_path))
        
        if self.save_weights_path:
            torch.save(self.model.state_dict(), self.save_weights_path)

    def predict(self, test_dataset: torch.utils.data.Dataset) -> torch.Tensor:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.model.eval()

        start_time = time.time()
        results = self.predict_inference(
            model=self.model,
            test_loader=test_loader
        )

        end_time = time.time()
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds.")
        return results