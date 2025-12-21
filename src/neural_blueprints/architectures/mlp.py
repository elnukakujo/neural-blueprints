import torch
import torch.nn as nn

from ..components.composite import FeedForwardNetwork
from ..config.architectures import MLPConfig
from ..utils import get_activation, get_input_projection, get_output_projection

import logging
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) architecture.
    
    Args:
        config (MLPConfig): Configuration for the MLP model.
    """
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()
        self.config = config

        if config.input_projection is not None:
            self.input_projection = get_input_projection(
                projection_config=config.input_projection
            )
            logger.info(f"Using input projection: {self.input_projection.__class__.__name__}")
            config.input_dim = self.input_projection.output_dim*len(self.input_projection.cardinalities)
        else:
            self.input_projection = None

        if config.output_projection is not None:
            config.output_dim = config.output_projection.latent_dim*len(config.output_projection.cardinalities)
            self.output_projection = get_output_projection(
                projection_config=config.output_projection
            )
            logger.info(f"Using output projection: {self.output_projection.__class__.__name__}")
        else:
            self.output_projection = None
        
        self.layers = nn.ModuleList()
        self.layers.append(FeedForwardNetwork(config))
        self.layers.append(get_activation(config.final_activation))
        self.layers = nn.Sequential(*self.layers)

        self.network = nn.Sequential(
            self.input_projection if config.input_projection is not None else nn.Identity(),
            self.layers
        )

    def blueprint(self) -> MLPConfig:
        print(self.network)
        return self.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            Output tensor after passing through the MLP.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(x)
            x = x.flatten(start_dim=1)
        pred = self.layers(x)
        if self.output_projection is not None:
            pred = self.output_projection(pred.unsqueeze(1))
        return pred