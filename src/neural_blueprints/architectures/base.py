import torch
import torch.nn as nn
from torchinfo import summary
from torchvista import trace_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

class BaseArchitecture(nn.Module):
    """Base class for neural network architectures.
    
    This class provides a template for building various neural network architectures.
    Subclasses should implement the `blueprint` method to return their configuration.
    """
    input_spec: Tuple | Dict
    output_spec: Tuple | Dict

    def _make_dummy_from_spec(
        self,
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
        assert hasattr(self, 'input_spec'), "Subclasses must define 'input_spec' attribute."

        if isinstance(self.input_spec, (list, tuple)):
            return torch.rand(batch_size, *self.input_spec)

        if isinstance(self.input_spec, dict):
            return {k: self._make_dummy_from_spec(v, batch_size) for k, v in self.input_spec.items()}
        
        raise ValueError("Input specification must be a tuple/list or a dictionary.")

    def blueprint(self, batch_size: Optional[int] = 64, with_graph: bool = True) -> None:
        """
        Print a summary of the model architecture.
        
        Args:
            batch_size (Optional[int]): The batch size to use for the summary. Default is 64.
            with_graph (bool): Whether to generate and display the model graph. Default is True.
        """
        dummy_input = self._make_dummy_from_spec(batch_size)
        print(summary(self, input_data=dummy_input))
        
        if with_graph:
            trace_model(self, inputs=dummy_input)
    
    def show_weights(
        self
    ) -> None:
        """
        Create interactive Plotly histograms of model weights grouped by component.
        """
        
        # Collect weights by component
        component_weights = defaultdict(list)
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            component = name.split('.')[0]
            
            # Flatten weights and add to component
            weights = param.data.cpu().flatten().numpy()
            component_weights[component].extend(weights)
        
        if not component_weights:
            print("No trainable parameters found!")
            return
        
        # Create subplots
        n_components = len(component_weights)
        n_cols = min(3, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(component_weights.keys()),
            vertical_spacing=0.2,
            horizontal_spacing=0.10
        )
        
        # Add histograms
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for idx, (component, weights) in enumerate(sorted(component_weights.items())):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=weights,
                    name=component,
                    marker_color=colors[idx % len(colors)],
                    opacity=0.7,
                    nbinsx=50,
                    showlegend=False
                ),
                row=row,
                col=col
            )
            
            # Update axes for this subplot
            fig.update_xaxes(title_text="Weight Value", row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title_text=f"{self.__class__.__name__} - Weight Distributions by components",
            showlegend=False,
            height=300 * n_rows,
            width=400 * n_cols,
            template="plotly_white"
        )
        
        fig.show()
    
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