import torch 
import torch.nn as nn

from ..components.composite import Generator, Discriminator
from ..config.architectures import GANConfig

class GAN(nn.Module):
    """A simple Generative Adversarial Network (GAN) architecture.
    
    Args:
        config (GANConfig): Configuration for the GAN model.
    """
    def __init__(self, config: GANConfig):
        super(GAN, self).__init__()
        self.config = config

        self.generator = Generator(
            layer_types=config.generator_config.layer_types,
            layer_configs=config.generator_config.layer_configs
        )

        self.discriminator = Discriminator(
            layer_types=config.discriminator_config.layer_types,
            layer_configs=config.discriminator_config.layer_configs
        )
        
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate data from latent representation.

        Args:
            z (torch.Tensor): Latent representation tensor.
        
        Returns:
            Generated data tensor.
        """
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate real or generated data.

        Args:
            x (torch.Tensor): Input data tensor.
        
        Returns:
            Discrimination result tensor.
        """
        return self.discriminator(x)