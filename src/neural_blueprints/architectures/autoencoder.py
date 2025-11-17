import torch
import torch.nn as nn

from ..components.composite import Encoder, Decoder
from ..config import AutoEncoderConfig, EncoderConfig, DecoderConfig

class AutoEncoder(nn.Module):
    """A simple AutoEncoder architecture composed of an encoder and a decoder.
    
    Args:
        config (AutoEncoderConfig): Configuration for the autoencoder model.
    """
    def __init__(self, config: AutoEncoderConfig):
        super(AutoEncoder, self).__init__()
        self.config = config

        self.encoder = Encoder(
            config = EncoderConfig(
                layer_types=config.encoder_layer_types,
                layer_configs=config.encoder_layer_configs,
            )
        )

        self.decoder = Decoder(
            config = DecoderConfig(
                layer_types=config.decoder_layer_types,
                layer_configs=config.decoder_layer_configs
            )
        )
        
    def blueprint(self) -> AutoEncoderConfig:
        print(self)
        return self.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Output tensor after passing through the autoencoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor into a latent representation.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            Latent representation tensor.
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent representation back to the original space.
        
        Args:
            z (torch.Tensor): Latent representation tensor.

        Returns:
            Reconstructed tensor.
        """
        return self.decoder(z)
    
class VariationalAutoEncoder(AutoEncoder):
    """A Variational AutoEncoder (VAE) architecture composed of an encoder and a decoder.
    
    Args:
        config (VariationalAutoEncoderConfig): Configuration for the VAE model.
    """
    def __init__(self, config: AutoEncoderConfig):
        super(VariationalAutoEncoder, self).__init__(config)

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
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input tensor into mean and log-variance tensors.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log-variance tensor.
        """
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, logvar