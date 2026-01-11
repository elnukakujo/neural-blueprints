from typing import Any, Tuple, Union, Dict, Optional
from pydantic import Field
from typing_extensions import TypeAlias

from .base import BaseType

TensorShape: TypeAlias = Tuple[int, ...]

SpecValue: TypeAlias = Union[
    TensorShape,
    Dict[str, Any], # Should be SpecValue recursively
]

UniModalInputSpec: TypeAlias = TensorShape

class MultiModalInputSpec(BaseType):
    """
    Example:
    {
        "tabular": (10,),
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

SingleOutputSpec: TypeAlias = TensorShape

class MultiOutputSpec(BaseType):
    """
    Example:
    {
        "classification": (10,),
        "regression": (1,),
    }
    """
    outputs: Dict[str, TensorShape]