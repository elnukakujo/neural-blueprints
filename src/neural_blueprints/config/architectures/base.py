from pydantic import BaseModel, model_validator
from typing import Optional
from abc import ABC

from ...types import (
    UniModalInputSpec,
    MultiModalInputSpec,
    SingleOutputSpec,
    MultiOutputSpec,
)
from ..components.composite.projections.input.base import BaseProjectionInputConfig
from ..components.composite.projections.output.base import BaseProjectionOutputConfig

class BaseArchitectureConfig(BaseModel, ABC):
    """Base configuration for neural architectures.

    Args:
        - input_dim (List[int]): Dimensions of the input features.
        - output_dim (Optional[List[int]]): Dimensions of the output features. If None, uses input_dim.
        - input_projection (Optional[BaseProjectionInputConfig]): Configuration for the input projection.
        - output_projection (Optional[BaseProjectionOutputConfig]): Configuration for the output projection.
        - dropout_p (Optional[float]): Dropout probability to apply in projections if not already set.
        - normalization (Optional[str]): Normalization type to apply in projections if not already set.
        - activation (Optional[str]): Activation function to apply in projections if not already set.
        - final_activation (Optional[str]): Final activation function to apply to the output.
    """
    
    input_spec: UniModalInputSpec | MultiModalInputSpec
    output_spec: SingleOutputSpec | MultiOutputSpec
    input_projection: Optional[BaseProjectionInputConfig] = None
    output_projection: Optional[BaseProjectionOutputConfig] = None
    dropout_p: Optional[float] = None
    normalization: Optional[str] = None
    activation: Optional[str] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if self.dropout_p is not None:
            self.input_projection.dropout_p = self.dropout_p if self.input_projection.normalization is None else self.input_projection.dropout_p
            self.output_projection.dropout_p = self.dropout_p if self.output_projection.normalization is None else self.output_projection.dropout_p
        if self.normalization is not None:
            self.input_projection.normalization = self.normalization if self.input_projection.normalization is None else self.input_projection.normalization
            self.output_projection.normalization = self.normalization if self.output_projection.normalization is None else self.output_projection.normalization
        if self.activation is not None:
            self.input_projection.activation = self.activation if self.input_projection.activation is None else self.input_projection.activation
            self.output_projection.activation = self.activation if self.output_projection.activation is None else self.output_projection.activation
        return self