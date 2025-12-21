import torch
import torch.nn as nn

from ...config.components.core import DropoutLayerConfig

import logging
logger = logging.getLogger(__name__)

class DropoutLayer(nn.Module):
    """Dropout layer component.
    
    Args:
        p (float): Dropout probability.
    """
    def __init__(self, config: DropoutLayerConfig):
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p=config.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dropout layer.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            Output tensor after applying dropout.
        """
        return self.dropout(x)