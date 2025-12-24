from typing import Optional
from pydantic import BaseModel, model_validator

from ..components.composite import EncoderConfig, DecoderConfig
from ..components.composite.projections.input import TabularInputProjectionConfig
from ..components.composite.projections.output import TabularOutputProjectionConfig

    
class AutoEncoderConfig(BaseModel):
    """Configuration for an AutoEncoder architecture.
    
    Args:
        input_projection (Optional[TabularInputProjectionConfig]): Configuration for the input projection.
        output_projection (Optional[TabularOutputProjectionConfig]): Configuration for the output projection.
        encoder_config (EncoderConfig): Configuration for the encoder.
        decoder_config (DecoderConfig): Configuration for the decoder.
    """
    input_projection: Optional[TabularInputProjectionConfig] = None
    output_projection: Optional[TabularOutputProjectionConfig] = None
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig

    @model_validator(mode='after')
    def _validate(self):
        return self