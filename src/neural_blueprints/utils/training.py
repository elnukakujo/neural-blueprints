"""Training utilities"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from tqdm import tqdm


class Trainer:
    """
    Generic trainer for neural networks.
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, device: Optional[torch.device] = None):
        """
        Args:
            model: Neural network model
            optimizer: PyTorch optimizer
            criterion: Loss function
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test data"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Compute accuracy for classification
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        metrics = {"loss": total_loss / len(dataloader)}
        if total > 0:
            metrics["accuracy"] = 100.0 * correct / total
        
        return metrics
    
    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[torch.utils.data.DataLoader] = None,
              num_epochs: int = 10,
              callbacks: Optional[list] = None) -> Dict[str, list]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            callbacks: List of callback functions
            
        Returns:
            Dictionary with training history
        """
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                print(f"Val Loss: {val_metrics['loss']:.4f}", end="")
                
                if "accuracy" in val_metrics:
                    history["val_accuracy"].append(val_metrics["accuracy"])
                    print(f" | Val Accuracy: {val_metrics['accuracy']:.2f}%")
                else:
                    print()
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_metrics, val_metrics if val_loader else None)
        
        return history