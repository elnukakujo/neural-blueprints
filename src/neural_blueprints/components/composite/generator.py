from .decoder import Decoder
from ...config import GeneratorConfig

class Generator(Decoder):
    """Generator model specialized from Decoder for GAN architectures."""
    def __init__(self, config: GeneratorConfig):
        super(Generator, self).__init__(config)
        pass