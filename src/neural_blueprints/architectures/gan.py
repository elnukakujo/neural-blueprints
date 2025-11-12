import torch 
import torch.nn as nn

from ..components.composite import Generator, Discriminator
from ..config import GANConfig

class GAN(nn.Module):
    """A simple Generative Adversarial Network (GAN) architecture."""
    def __init__(self, config: GANConfig):
        super(GAN, self).__init__()
        self.generator_config = config.generator_config
        self.discriminator_config = config.discriminator_config

        self.generator = Generator(
            layer_types=self.generator_config.layer_types,
            layer_configs=self.generator_config.layer_configs
        )

        self.discriminator = Discriminator(
            layer_types=self.discriminator_config.layer_types,
            layer_configs=self.discriminator_config.layer_configs
        )

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)