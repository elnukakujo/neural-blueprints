from pydantic import model_validator
from .base import BaseProjectionConfig
from .....types import UniModalSpec, MultiModalSpec

class MultiModalProjectionConfig(BaseProjectionConfig):
    input_spec: UniModalSpec | MultiModalSpec
    output_spec: UniModalSpec | MultiModalSpec

    @model_validator(mode="after")
    def _validate(self):
        if isinstance(self.input_spec, UniModalSpec) and isinstance(self.output_spec, UniModalSpec):
            raise ValueError("For uni-modal specs, use the specific uni-modal projection config.")
        return self