from typing import Optional
from pydantic import model_validator

from .base import BaseProjectionConfig
from .....types import ModalDim, ProjectionDim

class MultiModalProjectionConfig(BaseProjectionConfig):
    """
    Configuration for Multi-Modal Projection components.

    Args:
        - input_modalities_dim (Optional[ModalDim]): Input modalities dimensions. If set, the projection will be configured from Multi-Modal to Projection space.
        - output_modalities_dim (Optional[ModalDim]): Output modalities dimensions. If set, the projection will be configured from Projection to Multi-Modal space.
        - projection_dim (ProjectionDim): Target projection dimension.
        - hidden_dims (Optional[List[int]]): List of hidden layer dimensions.
        - dropout_p (Optional[float]): Dropout probability.
        - normalization (Optional[str]): Type of normalization to use.
        - activation (Optional[str]): Activation function to use.

    Note:
        Only one of input_modalities_dim or output_modalities_dim should be set.
    """
    input_modalities_dim: Optional[ModalDim] = None
    output_modalities_dim: Optional[ModalDim] = None

    projection_dim: ProjectionDim

    @model_validator(mode="after")
    def _validate(self):
        if self.input_modalities_dim and self.output_modalities_dim:
            raise ValueError("Only one of input_modalities_dim or output_modalities_dim should be set.")
        return self