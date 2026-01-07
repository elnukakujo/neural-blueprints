from pydantic import BaseModel

class BaseProjectionOutputConfig(BaseModel):
    """
    Base configuration for output projections.

    This class serves as a base for specific output projection configurations.
    It can be extended to include common attributes or methods shared across
    different output projection types.
    """
    pass