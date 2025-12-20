from pydantic import BaseModel, model_validator

class NormalizationConfig(BaseModel):
    """Configuration for a normalization layer.
    
    Args:
        norm_type (str): Type of normalization. Options: 'batchnorm1d', 'batchnorm2d', 'layernorm'.
        num_features (int): Number of features/channels for the normalization layer.
    """
    norm_type: str | None
    num_features: int

    @model_validator(mode='after')
    def _validate(self):
        if self.norm_type is not None and self.norm_type.lower() not in ('batchnorm1d', 'batchnorm2d', 'layernorm'):
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Supported types: 'batchnorm1d', 'batchnorm2d', 'layernorm'")
        if self.num_features <= 0:
            raise ValueError("num_features must be a positive integer")
        return self