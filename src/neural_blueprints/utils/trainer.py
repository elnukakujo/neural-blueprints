import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
logger = logging.getLogger(__name__)

from .metrics import get_criterion, get_optimizer
from .device import get_device
from .visualize import curve_plot
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
        self.optimizer = get_optimizer(config.optimizer, model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.criterion = get_criterion(config.criterion)
        self.training_type = config.training_type.lower()
        self.save_weights_path = config.save_weights_path
        self.batch_size = config.batch_size

        if self.save_weights_path is not None and not os.path.exists(os.path.dirname(self.save_weights_path)):
            os.makedirs(os.path.dirname(self.save_weights_path), exist_ok=False)
        else:
            print(f"Directory {os.path.dirname(self.save_weights_path)} already exists. Weights file may be overwritten.")

        self.best_val_loss = float('inf')
        os.makedirs(TEMP_DIR, exist_ok=True)
        self.model_path = os.path.join(TEMP_DIR, "best_model.pth")

        logger.info(f"Trainer initialized on device: {self.device}")

    def train_epoch_masked_label(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch, mask in train_loader:
            self.optimizer.zero_grad()
            X_batch, y_batch, mask = X_batch.to(self.device), y_batch.to(self.device), mask.to(self.device)
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred = y_pred, y_true = y_batch, mask = mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss
    
    def train_epoch_reconstruction(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred = y_pred, y_true = X_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss
    
    def train_epoch_label(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred = y_pred, y_true = y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss
    
    def evaluate_masked_label(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, mask in val_loader:
                X_batch, y_batch, mask = X_batch.to(self.device), y_batch.to(self.device), mask.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred = y_pred, y_true = y_batch, mask = mask)
                total_loss += loss.item() * X_batch.size(0)
            avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss
    
    def evaluate_reconstruction(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred = y_pred, y_true = X_batch)
                total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss
    
    def evaluate_label(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred = y_pred, y_true = y_batch)
                total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        if self.training_type == 'reconstruction':
            return self.train_epoch_reconstruction(train_loader)
        elif self.training_type == 'masked_label':
            return self.train_epoch_masked_label(train_loader)
        elif self.training_type == 'label':
            return self.train_epoch_label(train_loader)
        else:
            raise ValueError(f"Unknown training type: {self.training_type}")

    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> float:
        if self.training_type == 'reconstruction':
            val_loss = self.evaluate_reconstruction(val_loader)
        elif self.training_type == 'masked_label':
            val_loss = self.evaluate_masked_label(val_loader)
        elif self.training_type == 'label':
            val_loss = self.evaluate_label(val_loader)
        else:
            raise ValueError(f"Unknown training type: {self.training_type}")
        if self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.model_path)
        return val_loss
    
    def train(self, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, epochs: int, visualize: bool = True):
        train_losses = []
        val_losses = []

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
            train_losses.append(self.train_epoch(train_loader))
            val_losses.append(self.evaluate(val_loader))
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

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

        self.model.load_state_dict(torch.load(self.model_path))
        
        if self.save_weights_path:
            torch.save(self.model.state_dict(), self.save_weights_path)
    
    def predict(self, X: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            y_pred = self.model(X)
            if self.training_type == 'reconstruction':
                loss = self.criterion(y_pred = y_pred, y_true = X)
            elif self.training_type == 'masked_label':
                loss = self.criterion(y_pred = y_pred, y_true = y, mask=mask)
            elif self.training_type == 'label':
                loss = self.criterion(y_pred = y_pred, y_true = y)
            else:
                raise ValueError(f"Unknown training type: {self.training_type}")
        return y_pred, loss.item()