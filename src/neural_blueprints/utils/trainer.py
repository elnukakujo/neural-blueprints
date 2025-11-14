import torch
import torch.nn as nn
from tqdm import tqdm
import os

from .metrics import get_criterion
from .device import get_device
from .visualize import curve_plot

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: str|nn.Module,
            device: torch.device = None,
            is_reconstruction: bool = False,
            save_weights_path: str = None
        ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion if isinstance(criterion, nn.Module) else get_criterion(criterion)
        self.device = device if device is not None else get_device()
        self.is_reconstruction = is_reconstruction
        self.save_weights_path = save_weights_path if save_weights_path is not None else None

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)

            if self.is_reconstruction:
                loss = self.criterion(y_pred = y_pred, y_true = X_batch)
            else:
                loss = self.criterion(y_pred = y_pred, y_true = y_batch)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss
    
    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                if self.is_reconstruction:
                    loss = self.criterion(y_pred = y_pred, y_true = X_batch)
                else:
                    loss = self.criterion(y_pred = y_pred, y_true = y_batch)
                total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(val_loader.dataset)
        if self.best_val_loss > avg_loss:
            self.best_val_loss = avg_loss
            torch.save(self.model.state_dict(), "best_model.pth")
        return avg_loss
    
    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int, visualize: bool = True):
        train_losses = []
        val_losses = []
        self.best_val_loss = float('inf')

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

        self.model.load_state_dict(torch.load("best_model.pth"))
        os.remove("best_model.pth")
        
        if self.save_weights_path:
            torch.save(self.model.state_dict(), self.save_weights_path)
    
    def predict(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            y_pred = self.model(X)
            if self.is_reconstruction:
                loss = self.criterion(y_pred = y_pred, y_true = X)
            else:
                loss = self.criterion(y_pred = y_pred, y_true = y)
        return y_pred.cpu(), loss.item()