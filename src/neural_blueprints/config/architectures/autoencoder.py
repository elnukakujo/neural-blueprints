from pydantic import BaseModel, model_validator

from ..components.composite import EncoderConfig, DecoderConfig
    
class AutoEncoderConfig(BaseModel):
    """Configuration for an AutoEncoder architecture.
    
    Args:
        encoder_config (EncoderConfig): Configuration for the encoder.
        decoder_config (DecoderConfig): Configuration for the decoder.
    """

    encoder_config: EncoderConfig
    decoder_config: DecoderConfig

    @model_validator(mode='after')
    def _validate(self):
        return self