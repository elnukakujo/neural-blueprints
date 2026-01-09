from ..components.composite import EncoderConfig, DecoderConfig
from .base import BaseArchitectureConfig

class AutoEncoderConfig(BaseArchitectureConfig):
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig