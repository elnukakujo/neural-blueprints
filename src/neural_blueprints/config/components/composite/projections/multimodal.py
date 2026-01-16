from .base import BaseProjectionConfig
from .....types import MultiModalInputSpec, MultiOutputSpec, SingleOutputSpec

class MultiModalInputProjectionConfig(BaseProjectionConfig):
    input_spec: MultiModalInputSpec
    output_spec: SingleOutputSpec | MultiOutputSpec