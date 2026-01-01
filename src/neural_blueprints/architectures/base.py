import torch
import torch.nn as nn
from torchinfo import summary
from torchvista import trace_model
from typing import Optional, List, Tuple

class BaseArchitecture(nn.Module):
    """Base class for neural network architectures.
    
    This class provides a template for building various neural network architectures.
    Subclasses should implement the `blueprint` method to return their configuration.
    """
    input_dim: List[int] | Tuple[int, ...]
    output_dim: List[int] | Tuple[int, ...]

    def blueprint(self, batch_size: Optional[int] = 64) -> None:
        """
        Print a summary of the model architecture.
        
        Args:
            batch_size (Optional[int]): The batch size to use for the summary. Default is 64.
        """
        assert hasattr(self, 'input_dim'), "Subclasses must define 'input_dim' attribute."
        input_size = (batch_size, *self.input_dim)
        print(summary(self, input_size=input_size))

        trace_model(self, inputs=torch.rand(*input_size))
    
    def forward(self):
        raise NotImplementedError("Subclasses must implement the blueprint method.")
    
    def freeze(self):
        """Freeze the model parameters to prevent them from being updated during training."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze the model parameters to allow them to be updated during training."""
        for param in self.parameters():
            param.requires_grad = True

class EncoderArchitecture(BaseArchitecture):
    """Base class for encoder architectures."""
    def __init__(self):
        super(EncoderArchitecture, self).__init__()
    
    def encode(self):
        """
        Encode the input data.

        Returns:
            Encoded representation of the input data.
        """
        raise NotImplementedError("Subclasses must implement the encode method.")
    
class DecoderArchitecture(BaseArchitecture):
    """Base class for decoder architectures."""
    def __init__(self):
        super(DecoderArchitecture, self).__init__()
    
    def decode(self):
        """
        Decode the encoded data.

        Returns:
            Decoded representation of the input data.
        """
        raise NotImplementedError("Subclasses must implement the decode method.")

class EncoderDecoderArchitecture(BaseArchitecture):
    """Base class for encoder-decoder architectures."""
    def __init__(self):
        super(EncoderDecoderArchitecture, self).__init__()
    
    def encode(self):
        """
        Encode the input data.

        Returns:
            Encoded representation of the input data.
        """
        raise NotImplementedError("Subclasses must implement the encode method.")
    
    def decode(self):
        """
        Decode the encoded data.

        Returns:
            Decoded representation of the input data.
        """
        raise NotImplementedError("Subclasses must implement the decode method.")