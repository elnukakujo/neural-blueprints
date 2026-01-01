import torch.nn as nn

class BaseComposite(nn.Module):
    """Base class for composite components made up of multiple sub-components.
    
    This class provides a template for building composite components that consist of multiple
    sub-components.
    """
    input_dim: list[int]
    output_dim: list[int]

    def forward(self):
        raise NotImplementedError("Subclasses must implement the blueprint method.")