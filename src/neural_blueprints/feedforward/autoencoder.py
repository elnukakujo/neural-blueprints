"""AutoEncoder and Variational AutoEncoder"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .mlp import FeedForwardNN
from ..base import NeuralNetworkBase


class AutoEncoder(NeuralNetworkBase):
    """
    Standard AutoEncoder with symmetric encoder-decoder architecture.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 latent_dim: int, activation: str = "relu"):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden dimensions for encoder (decoder is mirrored)
            latent_dim: Bottleneck/latent dimension
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if i < len(encoder_dims) - 2:
                encoder_layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode"""
        z = self.encode(x)
        return self.decode(z)


class VariationalAE(AutoEncoder):
    """
    Variational AutoEncoder with reparameterization trick.
    Inherits from AutoEncoder and adds stochastic latent sampling.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 latent_dim: int, activation: str = "relu"):
        super().__init__(input_dim, hidden_dims, latent_dim, activation)
        
        # Replace encoder output with mean and logvar layers
        encoder_dims = [input_dim] + hidden_dims
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            encoder_layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, mu, and logvar"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from latent space"""
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        return self.decode(z)
    
    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor, 
                     beta: float = 1.0) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence)
        
        Args:
            beta: Weight for KL term (beta-VAE)
        """
        recon_loss = nn.functional.mse_loss(reconstruction, x.view_as(reconstruction), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss