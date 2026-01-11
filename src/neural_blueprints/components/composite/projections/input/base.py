from typing import Tuple

from ... import BaseComposite

class BaseInputProjection(BaseComposite):
    output_dim: Tuple[int, ...] = None