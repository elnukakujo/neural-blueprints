import pandas as pd
import numpy as np
import torch
from typing import Any, List 

def infer_types(data: Any) -> List[str]:
    """
    Infers column types and returns a DataFrame with correct dtypes.
    
    Types returned: 'categorical', 'discrete', 'continuous'.
    
    Args:
        data: The input data (pd.DataFrame, pd.Series, list, or torch.Tensor)
    
    Returns:
        types_list: List of inferred types for each column.
    """
    # Convert input to DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series) or isinstance(data, list) or isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, torch.Tensor):
        arr = data.cpu().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr)
    else:
        raise TypeError(f"Unsupported data type for type inference: {type(data)}")

    types_list = []
    
    for col in df.columns:
        col_data = df[col].to_numpy()
        if np.issubdtype(col_data.dtype, np.number):
            # Detect discrete vs continuous
            mask_int = np.isclose(col_data, np.round(col_data), atol=1e-8)
            if np.all(mask_int):
                types_list.append("int32")
            else:
                types_list.append("float32")
        else:
            types_list.append("category")
    
    return types_list