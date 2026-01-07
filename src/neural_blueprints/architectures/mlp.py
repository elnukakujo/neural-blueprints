import torch
import torch.nn as nn

from .base import BaseArchitecture
from ..components.composite import FeedForwardNetwork
from ..config.architectures import MLPConfig

import logging
logger = logging.getLogger(__name__)

class MLP(BaseArchitecture):
    """A simple Multi-Layer Perceptron (MLP) architecture.
    
    Args:
        config (MLPConfig): Configuration for the MLP model.
    """
    def __init__(self, config: MLPConfig):
        from ..utils import get_activation, get_input_projection, get_output_projection
        super(MLP, self).__init__()
        self.config = config

        if config.input_projection is not None:
            self.input_projection = get_input_projection(
                projection_config=config.input_projection
            )
            self.input_dim = self.input_projection.input_dim
            logger.info(f"Using input projection: {self.input_projection.__class__.__name__}")
            config.input_dim = self.input_projection.output_dim[-1]
        else:
            self.input_dim = config.input_dim
            self.input_projection = None

        if config.output_projection is not None:
            config.output_dim = config.output_projection.input_dim[-1]
            self.output_projection = get_output_projection(
                projection_config=config.output_projection
            )
            self.output_dim = self.output_projection.output_dim
            logger.info(f"Using output projection: {self.output_projection.__class__.__name__}")
        else:
            self.output_dim = config.output_dim
            self.output_projection = None
        
        self.layers = nn.ModuleList()
        self.layers.append(FeedForwardNetwork(config))
        self.layers.append(get_activation(config.final_activation))
        self.layers = nn.Sequential(*self.layers)

        self.network = nn.Sequential(
            self.input_projection if config.input_projection is not None else nn.Identity(),
            self.layers,
            self.output_projection if config.output_projection is not None else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            Output tensor after passing through the MLP.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(x)
        pred = self.layers(x)
        if self.output_projection is not None:
            pred = self.output_projection(pred)
        return pred