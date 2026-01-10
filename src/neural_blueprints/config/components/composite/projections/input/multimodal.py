from typing import List, Optional

from .base import BaseProjectionInputConfig

class MultiModalInputProjectionConfig(BaseProjectionInputConfig):
    tabular_cardinalities: Optional[List[int]] = None
    representation_input_dim: Optional[List[int]] = None