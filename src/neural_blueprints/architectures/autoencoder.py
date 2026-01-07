import torch
import torch.nn as nn

from .base import EncoderDecoderArchitecture
from ..components.composite import Encoder, Decoder
from ..config.architectures import AutoEncoderConfig

import logging
logger = logging.getLogger(__name__)

class AutoEncoder(EncoderDecoderArchitecture):
    """A simple AutoEncoder architecture composed of an encoder and a decoder.
    
    Args:
        config (AutoEncoderConfig): Configuration for the autoencoder model.
    """
    def __init__(self, config: AutoEncoderConfig):
        from ..utils import get_input_projection, get_output_projection
        super(AutoEncoder, self).__init__()
        self.config = config

        self.encoder = Encoder(
            config=config.encoder_config
        )

        self.decoder = Decoder(
            config=config.decoder_config
        )

        if config.input_projection is not None:
            self.input_projection = get_input_projection(
                projection_config=config.input_projection
            )
            self.input_dim = self.input_projection.input_dim
            logger.info(f"Using input projection: {self.input_projection.__class__.__name__}")
        else:
            self.input_dim = self.encoder.input_dim
            self.input_projection = None

        if config.output_projection is not None:
            self.output_projection = get_output_projection(
                projection_config=config.output_projection
            )
            self.output_dim = self.output_projection.output_dim
            logger.info(f"Using output projection: {self.output_projection.__class__.__name__}")
        else:
            self.output_dim = self.decoder.output_dim
            self.output_projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Output tensor after passing through the autoencoder.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(x)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if self.output_projection is not None:
            decoded = self.output_projection(decoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor into a latent representation.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            Latent representation tensor.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(x)
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent representation back to the original space.
        
        Args:
            z (torch.Tensor): Latent representation tensor.

        Returns:
            Reconstructed tensor.
        """
        decoded = self.decoder(z)

        if self.output_projection is not None:
            decoded = self.output_projection(decoded)

        return decoded
    
class VariationalAutoEncoder(AutoEncoder):
    """A Variational AutoEncoder (VAE) architecture composed of an encoder and a decoder.
    
    Args:
        config (VariationalAutoEncoderConfig): Configuration for the VAE model.
    """
    def __init__(self, config: AutoEncoderConfig):
        super(VariationalAutoEncoder, self).__init__(config)

        self.logvar_layer = nn.Linear(
            in_features=self.encoder.layer_configs[-1].output_dim,
            out_features=self.encoder.layer_configs[-1].output_dim//2
        )

        self.mu_layer = nn.Linear(
            in_features=self.encoder.layer_configs[-1].output_dim,
            out_features=self.encoder.layer_configs[-1].output_dim//2
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log-variance tensor.

        Returns:
            Sampled tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            decoded (torch.Tensor): Reconstructed tensor.
            mu (torch.Tensor): Mean tensor from the encoder.
            logvar (torch.Tensor): Log-variance tensor from the encoder.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(x)

        z = self.encoder(x)
        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        if self.output_projection is not None:
            decoded = self.output_projection(decoded)
        return decoded, mu, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input tensor into mean and log-variance tensors.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log-variance tensor.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(x)
            
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, logvar