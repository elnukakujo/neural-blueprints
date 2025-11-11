"""Base Neural Network Class"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json


class NeuralNetworkBase(nn.Module, ABC):
    """
    Abstract base class for all neural network architectures.
    Provides common functionality for training, saving, and loading models.
    """
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        pass
    
    def train_step(self, batch: tuple, optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """
        Single training step
        
        Args:
            batch: Tuple of (inputs, targets)
            optimizer: PyTorch optimizer
            criterion: Loss function
            
        Returns:
            Loss value
        """
        self.train()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader, 
                 criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate model on validation/test data
        
        Args:
            dataloader: DataLoader for evaluation
            criterion: Loss function
            
        Returns:
            Dictionary with metrics
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self(inputs)
                loss = criterion(outputs, targets)
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
    
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
            "metadata": metadata or {}
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("metadata", {})
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)