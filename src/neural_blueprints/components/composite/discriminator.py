from .encoder import Encoder
from ...config.components.composite import DiscriminatorConfig

class Discriminator(Encoder):
    """Discriminator model specialized from Encoder for GAN architectures.
    
    Args:
        config (DiscriminatorConfig): Configuration for the discriminator model.
    """
    def __init__(self, config: DiscriminatorConfig):
        super(Discriminator, self).__init__(config)
        pass