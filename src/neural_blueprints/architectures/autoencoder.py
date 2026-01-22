import torch
import torch.nn as nn

from .base import AutoencoderArchitecture
from ..config.architectures import AutoEncoderConfig
from ..config.components.composite import EncoderConfig, DecoderConfig
from ..config.components.core import DenseLayerConfig
from ..utils import get_block

import logging
logger = logging.getLogger(__name__)

class AutoEncoder(AutoencoderArchitecture):
    """A simple AutoEncoder architecture composed of an encoder and a decoder.
    
    Args:
        config (AutoEncoderConfig): Configuration for the autoencoder model.
    """
    def __init__(self, config: AutoEncoderConfig):
        from ..utils import get_projection
        super(AutoEncoder, self).__init__()
        self.input_spec = config.input_spec
        self.output_spec = config.output_spec
        self.type = "autoencoder"

        encoder_layers_dim = config.encoder_layers
        decoder_layers_dim = config.decoder_layers
        latent_dim = config.latent_dim
        
        encoder_attention_layers = config.encoder_attention_layers
        decoder_attention_layers = config.decoder_attention_layers
        latent_attention = config.latent_attention
        
        symmetric = config.symmetric

        if symmetric:
            if encoder_layers_dim is not None:
                decoder_layers_dim = encoder_layers_dim[::-1]
                decoder_attention_layers = encoder_attention_layers[::-1] if encoder_attention_layers else None
            elif decoder_layers_dim is not None:
                encoder_layers_dim = decoder_layers_dim[::-1]
                encoder_attention_layers = decoder_attention_layers[::-1] if decoder_attention_layers else None
            else:
                raise ValueError("For symmetric autoencoder, either encoder_layers or decoder_layers must be provided.")


        activation = config.activation
        normalization = config.normalization
        dropout_p = config.dropout_p
        final_activation = config.final_activation

        if config.input_projection is not None:
            self.input_projection = get_projection(
                projection_config=config.input_projection
            )
            logger.info(f"Using input projection: {self.input_projection.__class__.__name__}")
        else:
            self.input_projection = None

        self.encoder = get_block(
            EncoderConfig(
                layers_dim=encoder_layers_dim,
                latent_dim=latent_dim,
                attention_layers=encoder_attention_layers,
                normalization=normalization,
                activation=activation,
                dropout_p=dropout_p,
                final_activation=activation
            )
        )

        if config.latent_attention is not None:
            self.latent_attention_layer = nn.Sequential(
                *[
                    nn.TransformerEncoderLayer(
                        d_model=latent_dim,
                        nhead=8,
                        dim_feedforward=latent_dim*4,
                        dropout=dropout_p,
                        batch_first=True,
                        activation=activation
                    ) for _ in range(latent_attention)
                ]
            )
        else:
            self.latent_attention_layer = None

        self.decoder = get_block(
            DecoderConfig(
                layers_dim=decoder_layers_dim,
                latent_dim=latent_dim,
                attention_layers=decoder_attention_layers,
                normalization=normalization,
                activation=activation,
                dropout_p=dropout_p,
                final_activation=final_activation if not config.output_projection else activation
            )
        )

        if config.output_projection is not None:
            self.output_projection = get_projection(
                projection_config=config.output_projection
            )
            logger.info(f"Using output projection: {self.output_projection.__class__.__name__}")
        else:
            self.output_projection = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            Output tensor after passing through the autoencoder.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(inputs)
        else:
            x = inputs

        encoded = self.encoder(x)

        if self.latent_attention_layer is not None:
            encoded = self.latent_attention_layer(encoded)

        decoded = self.decoder(encoded)

        if self.output_projection is not None:
            decoded = self.output_projection(decoded)
        return decoded
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor into a latent representation.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            Latent representation tensor.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(inputs)
        else:
            x = inputs
        z = self.encoder(x)

        if self.latent_attention_layer is not None:
            z = self.latent_attention_layer(z)
        return z
    
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
        self.type = "vae"

        latent_dim = config.latent_dim

        self.logvar_layer = get_block(
            DenseLayerConfig(
                input_dim=[latent_dim],
                output_dim=[latent_dim],
                normalization=None,
                activation=None,
                dropout_p=None
            )
        )

        self.mu_layer = get_block(
            DenseLayerConfig(
                input_dim=[latent_dim],
                output_dim=[latent_dim],
                normalization=None,
                activation=None,
                dropout_p=None
            )
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

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            decoded (torch.Tensor): Reconstructed tensor.
            mu (torch.Tensor): Mean tensor from the encoder.
            logvar (torch.Tensor): Log-variance tensor from the encoder.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(inputs)
        else:
            x = inputs

        z = self.encoder(x)
        
        if self.latent_attention_layer is not None:
            z = self.latent_attention_layer(z)

        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        z = self.reparameterize(mu, logvar)

        decoded = self.decoder(z)
        if self.output_projection is not None:
            decoded = self.output_projection(decoded)
        return decoded, mu, logvar

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input tensor into mean and log-variance tensors.

        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log-variance tensor.
        """
        if self.input_projection is not None:
            x, _ = self.input_projection(inputs)
        else:
            x = inputs
            
        z = self.encoder(x)

        if self.latent_attention_layer is not None:
            z = self.latent_attention_layer(z)

        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        return mu, logvar