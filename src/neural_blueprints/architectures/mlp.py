import torch
import torch.nn as nn

from .base import BaseArchitecture
from ..components.composite import FeedForwardNetwork
from ..config.architectures import MLPConfig
from ..config.components.composite import FeedForwardNetworkConfig

import logging
logger = logging.getLogger(__name__)

class MLP(BaseArchitecture):
    """A simple Multi-Layer Perceptron (MLP) architecture.
    
    Args:
        config (MLPConfig): Configuration for the MLP model.
    """
    def __init__(self, config: MLPConfig):
        from ..utils import get_activation, get_projection
        super(MLP, self).__init__()
        self.config = config

        self.input_spec = config.input_spec
        self.output_spec = config.output_spec

        if config.input_projection is not None:
            self.input_projection = get_projection(
                projection_config=config.input_projection
            )
            logger.info(f"Using input projection: {self.input_projection.__class__.__name__}")
        else:
            assert isinstance(self.input_spec, tuple), f"Input spec must be a valid 1D shaped tuple if no input projection is provided but got: {self.input_spec}."
            self.input_projection = None

        if config.output_projection is not None:
            if config.final_activation:
                config.output_projection.final_activation = config.final_activation
            self.output_projection = get_projection(
                projection_config=config.output_projection
            )
            logger.info(f"Using output projection: {self.output_projection.__class__.__name__}")
        else:
            assert isinstance(self.output_spec, tuple), f"Output spec must be a valid 1D shaped tuple if no output projection is provided but got: {self.output_spec}."
            self.output_projection = None
        
        self.layers = nn.ModuleList()
        self.layers.append(FeedForwardNetwork(
            FeedForwardNetworkConfig(
                input_dim=self.input_spec if not self.input_projection else self.input_projection.output_dim,
                hidden_dims=config.hidden_dims,
                output_dim=self.output_spec if not self.output_projection else self.output_projection.input_dim,
                normalization=config.normalization,
                activation=config.activation,
                dropout_p=config.dropout_p
            )
        ))
        if not self.output_projection:
            self.layers.append(get_activation(config.final_activation))
        self.layers = nn.Sequential(*self.layers)

        self.network = nn.Sequential(
            self.input_projection if config.input_projection is not None else nn.Identity(),
            self.layers[:-1],
            self.output_projection if config.output_projection is not None else self.layers[-1]
        )

    def forward(self, inputs):
        if self.input_projection is not None:
            inputs, _ = self.input_projection(inputs)
        pred = self.layers(inputs)
        if self.output_projection is not None:
            pred = self.output_projection(pred)
        return pred