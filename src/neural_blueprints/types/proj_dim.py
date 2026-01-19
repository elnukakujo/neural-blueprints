from typing import Any, Tuple, Union, Dict, Optional
from pydantic import Field
from typing_extensions import TypeAlias

from .base import BaseType

TensorShape: TypeAlias = Tuple[int, ...]

SpecValue: TypeAlias = Union[
    TensorShape,
    Dict[str, Any], # Should be SpecValue recursively
]

SingleProjectionDim: TypeAlias = Tuple[int, ...]

class MultiProjectionDim(BaseType, total=False):
    """
    Example:
    {
        "tabular": (128,),
        "representation": {
            "0-4*": (256,), # Concatenate modalities 0 to 4 and train on a single projection of output dimension (256,)
            "5+7*": (32,),  # Concatenate modalities 5 and 7 and train on a single projection of output dimension (32,)
            "6+8": (64,),   # Train separate projections for modalities 6 and 8 each of output dimension (64,) and do not concatenate their outputs
        }
    }
    """
    tabular: Optional[SpecValue] = Field(default=None)
    image: Optional[SpecValue] = Field(default=None)
    text: Optional[SpecValue] = Field(default=None)
    audio: Optional[SpecValue] = Field(default=None)
    representation: Optional[SpecValue] = Field(default=None)

ProjectionDim: TypeAlias = SingleProjectionDim | MultiProjectionDim