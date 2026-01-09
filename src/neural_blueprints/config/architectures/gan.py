from .base import BaseArchitectureConfig
from ..components.composite import GeneratorConfig, DiscriminatorConfig

class GANConfig(BaseArchitectureConfig):

    generator_config: GeneratorConfig
    discriminator_config: DiscriminatorConfig