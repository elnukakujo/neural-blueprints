"""Configuration management"""
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class ModelConfig:
    """Configuration for neural network models"""
    
    # Model architecture
    model_type: str = "feedforward"  # feedforward, cnn, rnn, transformer, etc.
    
    # Common parameters
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    
    # Transformer-specific
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    
    # CNN-specific
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    
    # RNN-specific
    hidden_size: int = 512
    num_rnn_layers: int = 2
    bidirectional: bool = False
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 0.01
    
    # Optimizer
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: Optional[str] = None  # cosine, step, plateau
    
    # Misc
    seed: int = 42
    device: str = "cuda"
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"ModelConfig({', '.join(items)})"