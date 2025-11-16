import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..utils.types import infer_types

class TabularPreprocessor:
    def __init__(self, with_masking: bool = False, normalize_discrete: bool = False):
        self.with_masking = with_masking
        self.normalize_discrete = normalize_discrete

    def run(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        # Determine categorical and continuous columns
        data_types = infer_types(data)
        data_dtypes = {col: dtype for col, dtype in zip(data.columns, data_types)}
        data = data.astype(data_dtypes)

        discrete_features = []
        continuous_features = []

        for col in data.columns:
            if data[col].dtype.name == 'category':
                discrete_features.append(col)
            elif data[col].dtype.name in ['int64', 'int32'] and data[col].nunique() < 50:
                discrete_features.append(col)
            elif data[col].dtype.name in ['float64', 'float32', 'int64', 'int32']:
                continuous_features.append(col)

        # Encode categorical columns
        self.label_encoders = {}

        NAN_TOKEN = "<NAN>"
        MASK_TOKEN = "<MASK>" if self.with_masking else None

        for col in discrete_features:
            column_data = data[col].astype(str)
            column_data = column_data.replace(['nan', 'NaN', 'NAN', '?', '', ' '], NAN_TOKEN)
            
            unique_vals = column_data.unique().tolist()
            if NAN_TOKEN in unique_vals:
                unique_vals.remove(NAN_TOKEN)
            if MASK_TOKEN is not None and MASK_TOKEN in unique_vals:
                unique_vals.remove(MASK_TOKEN)

            unique_vals = sorted(unique_vals)
            
            if MASK_TOKEN is not None:
                ordered_vals = [NAN_TOKEN, MASK_TOKEN] + unique_vals
            else:
                ordered_vals = [NAN_TOKEN] + unique_vals
            
            le = LabelEncoder()
            le.classes_ = np.array(ordered_vals)
            data[col] = le.transform(column_data)
            self.label_encoders[col] = le
            
        # 2. Handle continuous columns
        self.num_scaler = MinMaxScaler()
        if self.normalize_discrete:
            data = pd.DataFrame(self.num_scaler.fit_transform(data), columns=data.columns)
        else:
            data[continuous_features] = self.num_scaler.fit_transform(data[continuous_features].values)

        return data, discrete_features, continuous_features