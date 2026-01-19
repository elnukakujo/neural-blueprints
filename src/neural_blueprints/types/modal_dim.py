from typing import Any, Tuple, Union, Dict, Optional
from pydantic import Field
from typing_extensions import TypeAlias

from .base import BaseType

TensorShape: TypeAlias = Tuple[int, ...]

CardinalitiesType: TypeAlias = Tuple[int, ...]

SpecValue: TypeAlias = Union[
    TensorShape | CardinalitiesType,
    Dict[str, Any], # Should be SpecValue recursively
]

UniModalDim: TypeAlias = TensorShape | CardinalitiesType

class MultiModalDim(BaseType, total=False):
    """
    Example:
    {
        "tabular": (10, 4, 6),
        "representation": {
            "image": (512,),
            "text": (768,),
        }
    }
    """
    tabular: Optional[SpecValue] = Field(default=None)
    image: Optional[SpecValue] = Field(default=None)
    text: Optional[SpecValue] = Field(default=None)
    audio: Optional[SpecValue] = Field(default=None)
    representation: Optional[SpecValue] = Field(default=None)

ModalDim: TypeAlias = UniModalDim | MultiModalDim