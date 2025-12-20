from pydantic import BaseModel, model_validator

from ..composite import GeneratorConfig, DiscriminatorConfig

class GANConfig(BaseModel):
    """Configuration for a Generative Adversarial Network (GAN) architecture.
    
    Args:
        generator_config (GeneratorConfig): Configuration for the generator model.
        discriminator_config (DiscriminatorConfig): Configuration for the discriminator model.
    """

    generator_config: GeneratorConfig
    discriminator_config: DiscriminatorConfig

    @model_validator(mode='after')
    def _validate(self):
        return self