import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import logging
logger = logging.getLogger(__name__)

class TabularPreprocessor:
    """
    Preprocesses tabular datasets by encoding discrete features and scaling continuous features.

    - Discrete features are label encoded with the label 0 reserved for NaN values.
    - Continuous features are scaled to the range [0, 1] using Min-Max scaling and NaN values are replaced with -1.
    """
    def __init__(self):
        pass

    def discrete_encoding(self, discrete_features: list[str], original_df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Encode discrete features using label encoding with consistent class ordering.
        
        This method transforms discrete/categorical columns in the original dataframe and any
        additional dataframes provided via kwargs. It handles missing values by replacing them
        with a special token and ensures consistent encoding across all dataframes by using
        the unique values from the original dataframe as the reference.
        
        Args:
            discrete_features (list[str]) : List of column names containing discrete/categorical features to encode.
            original_df (pd.DataFrame) : The primary dataframe containing the discrete features to encode.
            **kwargs (dict[str, pd.DataFrame]) : Additional dataframes to apply the same encoding transformations to. The encoders learned from original_df are applied to these dataframes.
        
        Returns:
            original_df: The original_df with discrete features label-encoded.
            kwargs: The kwargs dictionary with all provided dataframes similarly encoded.
        """
        # Encode categorical columns
        self.label_encoders = {}

        NAN_TOKEN = "<NAN>"

        for col in discrete_features:
            column_data = original_df[col].astype(str).replace("nan", NAN_TOKEN)
            unique_vals = column_data.unique().tolist()

            if NAN_TOKEN in unique_vals:
                unique_vals.remove(NAN_TOKEN)
            
            ordered_vals = [NAN_TOKEN] + sorted(unique_vals)
            
            le = LabelEncoder()
            le.classes_ = np.array(ordered_vals)
            assert le.classes_[0] == NAN_TOKEN, f"First class must be the NAN_TOKEN for column {col}, got {le.classes_[0]}"

            original_df[col] = le.transform(column_data)
            original_df[col] = original_df[col].astype('category')

            for key, df in kwargs.items():
                df[col] = df[col].astype(str).replace("nan", NAN_TOKEN)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else NAN_TOKEN)
                kwargs[key][col] = le.transform(df[col])
                kwargs[key][col] = kwargs[key][col].astype('category')
            self.label_encoders[col] = le
                
        return original_df, kwargs
    
    def continuous_scaling(self, continuous_features: list[str], original_df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Scale continuous features using Min-Max scaling to the range [0, 1].

        This method scales continuous/numerical columns in the original dataframe and any
        additional dataframes provided via kwargs. It handles missing values by replacing them
        with a special token and ensures consistent scaling across all dataframes by using
        the scaler fitted on the original dataframe as the reference.

        Args:
            continuous_features (list[str]) : List of column names containing continuous/numerical features to scale.
            original_df (pd.DataFrame) : The primary dataframe containing the continuous features to scale.
            **kwargs (dict[str, pd.DataFrame]) : Additional dataframes to apply the same scaling transformations to. The scalers learned from original_df are applied to these dataframes.

        Returns:
            original_df: The original_df with continuous features scaled.
            kwargs: The kwargs dictionary with all provided dataframes similarly scaled.
        """
        # Scale numerical columns
        self.scalers = {}
        for col in continuous_features:
            scaler = MinMaxScaler()
            original_df[[col]] = scaler.fit_transform(original_df[[col]])
            original_df[col] = original_df[col].replace(np.nan, -1)  # Use -1 for NaNs
            original_df[col] = original_df[col].astype('float32')

            for key, df in kwargs.items():
                df[col] = scaler.transform(df[[col]])
                kwargs[key][col] = df[col].replace(np.nan, -1)  # Use -1 for NaNs
                kwargs[key][col] = df[col].astype('float32')

            self.scalers[col] = scaler
        
        return original_df, kwargs

    def run(self, original_df: pd.DataFrame, verbose: bool = True, **kwargs) -> tuple[tuple[pd.DataFrame, ...] | pd.DataFrame, list[str], list[str]]:
        """
        Run the preprocessing pipeline on the original dataframe and any additional dataframes.

        This method identifies discrete and continuous features in the original dataframe,
        applies discrete encoding and continuous scaling, and ensures consistent data types
        across all dataframes.

        Args:
            original_df (pd.DataFrame) : The primary dataframe to preprocess.
            verbose (bool) : Whether to log detailed information during preprocessing.
            **kwargs (dict[str, pd.DataFrame]) : Additional dataframes to preprocess similarly.

        Returns:
            datasets: A tuple containing the preprocessed original_df and any additional dataframes.
            discrete_features: List of identified discrete feature column names.
            continuous_features: List of identified continuous feature column names.
        """
        discrete_features = []
        continuous_features = []

        for col in original_df.columns:
            if original_df[col].dtype.name == 'category' or (original_df[col].dtype.name in ['int64', 'int32'] and original_df[col].nunique() < 50):
                discrete_features.append(col)
            elif original_df[col].dtype.name in ['float64', 'float32', 'int64', 'int32']:
                continuous_features.append(col)
            else:
                raise ValueError(f"Unsupported data type {original_df[col].dtype} for column {col}")

        if verbose:
            logger.info(f"Identified {len(discrete_features)} discrete features: {discrete_features}")
            logger.info(f"Identified {len(continuous_features)} continuous features: {continuous_features}")

        original_df, kwargs = self.discrete_encoding(discrete_features, original_df, **kwargs)
        original_df, kwargs = self.continuous_scaling(continuous_features, original_df, **kwargs)

        data = tuple([original_df] + list(kwargs.values())) if kwargs else original_df

        return data, discrete_features, continuous_features