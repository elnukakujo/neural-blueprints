import pandas as pd
import numpy as np
import torch
from typing import Any 

def infer_types(data: Any) -> dict[str, str]:
    """
    Infers column types and returns a list with correct dtypes.
    
    Types returned: 'category', 'int32', 'float32'.
    
    Args:
        data: The input data (pd.DataFrame, pd.Series, list, or torch.Tensor)
    
    Returns:
        types_dict: Dictionary of inferred types for each column name/index.
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

    types_dict = {}
    
    for col in df.columns:
        col_data = df[col].to_numpy()
        col_data = col_data[~pd.isnull(col_data)]  # Exclude NaNs

        if np.issubdtype(col_data.dtype, np.number):
            # Detect discrete vs continuous
            mask_int = np.isclose(col_data, np.round(col_data), atol=1e-12)
            if np.all(mask_int):
                types_dict[col] = "int32"
            else:
                types_dict[col] = "float32"
        elif np.issubdtype(col_data.dtype, np.str_) or np.issubdtype(col_data.dtype, np.object_) or np.issubdtype(col_data.dtype, np.bool_):
            types_dict[col] = "category"
        else:
            raise ValueError(f"Unrecognized data type in column {col}: {col_data.dtype}")
    
    return types_dict