import torch
import torch.nn as nn
from torchinfo import summary
from torchvista import trace_model
from typing import Optional, Tuple, Dict

class BaseArchitecture(nn.Module):
    """Base class for neural network architectures.
    
    This class provides a template for building various neural network architectures.
    Subclasses should implement the `blueprint` method to return their configuration.
    """
    input_spec: Tuple | Dict
    output_spec: Tuple | Dict
    type: str = "base"

    def _make_dummy_from_spec(
        self,
        spec: Tuple | Dict,
        batch_size: int
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Create a dummy tensor or dictionary of tensors based on the provided specification.

        Args:
            spec (Tuple | Dict): The input specification defining the shape of the input.
            batch_size (int): The batch size for the dummy tensor.
        
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: A dummy tensor or dictionary of tensors.
        """
        if spec is None:
            return None
        if isinstance(spec, (list, tuple)):
            return torch.rand(batch_size, *spec)

        if isinstance(spec, dict):
            return {k: self._make_dummy_from_spec(spec=v, batch_size=batch_size) for k, v in spec.items()}
        raise ValueError("Input specification must be a tuple/list or a dictionary.")

    def blueprint(self, batch_size: Optional[int] = 64, with_graph: bool = True) -> None:
        """
        Print a summary of the model architecture.
        
        Args:
            batch_size (Optional[int]): The batch size to use for the summary. Default is 64.
            with_graph (bool): Whether to generate and display the model graph. Default is True.
        """
        dummy_input = self._make_dummy_from_spec(spec = self.input_spec, batch_size=batch_size)

        print(summary(self, input_data={"inputs": dummy_input}))
        
        if with_graph:
            trace_model(self, inputs=dummy_input)
    
    def forward(self, inputs: torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the forward method.")
    
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
    
    def encode(self, inputs: torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> torch.Tensor | list[torch.Tensor]:
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
    
    def decode(self, encoded: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """
        Decode the encoded data.

        Returns:
            Decoded representation of the encoded data.
        """
        raise NotImplementedError("Subclasses must implement the decode method.")
    
class AutoencoderArchitecture(EncoderArchitecture, DecoderArchitecture):
    pass