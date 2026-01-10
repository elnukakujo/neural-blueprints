from typing import Tuple, Union, Dict, Optional
from typing_extensions import TypeAlias

from .base import BaseType

TensorShape: TypeAlias = Tuple[int, ...]

SpecValue: TypeAlias = Union[
    TensorShape,
    Dict[str, "SpecValue"],
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
    # MultiModal Case
    tabular: Optional[SpecValue] = None
    image: Optional[SpecValue] = None
    text: Optional[SpecValue] = None
    audio: Optional[SpecValue] = None
    representation: Optional[SpecValue] = None

    # UniModal Case
    inputs: Optional[TensorShape] = None

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