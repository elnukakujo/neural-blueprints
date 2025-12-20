from typing import List
from pydantic import BaseModel, model_validator
    
class AutoEncoderConfig(BaseModel):
    """Configuration for an AutoEncoder architecture.
    
    Args:
        encoder_layer_types (list of str): List of layer types for the encoder.
        encoder_layer_configs (list of BaseModel): List of configurations for each encoder layer.
        decoder_layer_types (list of str): List of layer types for the decoder.
        decoder_layer_configs (list of BaseModel): List of configurations for each decoder layer.
    """

    encoder_layer_types: List[str]
    encoder_layer_configs: List[BaseModel]
    decoder_layer_types: List[str]
    decoder_layer_configs: List[BaseModel]

    @model_validator(mode='after')
    def _validate(self):
        if len(self.encoder_layer_types) != len(self.encoder_layer_configs):
            raise ValueError("Length of encoder_layer_types must match length of encoder_layer_configs")
        if len(self.decoder_layer_types) != len(self.decoder_layer_configs):
            raise ValueError("Length of decoder_layer_types must match length of decoder_layer_configs")
        return self