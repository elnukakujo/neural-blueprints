import torch 
import torch.nn as nn

from ..components.composite import Generator, Discriminator
from ..config import GANConfig

class GAN(nn.Module):
    """A simple Generative Adversarial Network (GAN) architecture."""
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
    
    def blueprint(self) -> GANConfig:
        print(self)
        return self.config

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)