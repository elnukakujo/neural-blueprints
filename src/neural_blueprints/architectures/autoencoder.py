import torch
import torch.nn as nn

from ..components.composite import Encoder, Decoder
from ..config import AutoEncoderConfig, VariationalAutoEncoderConfig

class AutoEncoder(nn.Module):
    """A simple AutoEncoder architecture."""
    def __init__(self, config: AutoEncoderConfig):
        super(AutoEncoder, self).__init__()
        self.encoder_layer_types = config.encoder_config
        self.encoder_layer_configs = config.encoder_layer_configs
        self.decoder_layer_types = config.decoder_layer_types
        self.decoder_layer_configs = config.decoder_layer_configs


        self.encoder = Encoder(
            layer_types=self.encoder_layer_types,
            layer_configs=self.encoder_layer_configs
        )

        self.decoder = Decoder(
            layer_types=self.decoder_layer_types,
            layer_configs=self.decoder_layer_configs
        )

        if self.encoder.layers[-1].output_dim != self.decoder.layers[0].input_dim:
            raise ValueError("The output_dim of the last encoder layer must match the input_dim of the first decoder layer.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
class VariationalAutoEncoder(AutoEncoder):
    """A simple Variational AutoEncoder (VAE) architecture."""
    def __init__(self, config: VariationalAutoEncoderConfig):
        super(VariationalAutoEncoder, self).__init__(config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, logvar