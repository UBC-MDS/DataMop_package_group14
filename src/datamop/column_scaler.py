# Formula in this function is adapted from Scikit Learn documentation
# https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html

import pandas as pd
import numpy as np
import warnings

def column_scaler(data, column, method="minmax", new_min=0, new_max=1, inplace=True):
    """
    Scales the values of a specified column in a DataFrame.

    Parameters
    -----------
    data : pandas.DataFrame
        The DataFrame containing the column of interest for scaling.
    column: str
        The name of the numeric column to scale.
    method: str
        The method used for scaling. Options include:
            - `minmax`: Scales values between `new_min` and `new_max`, used as default method.
            - `standard`: Scales values with mean of 0 and standard deviation of 1.
    new_min: float
        The lower boundary value for min-max scaling. Default value is 0.
    new_max: float
        The upper boundary value for min-max scaling. Default value is 1. 
    inplace: bool
        If `True` the original column is replaced with new scaled values.
        If `False` the original column is retained and the new scaled column is 
        added to the dataframe with title `<column-name>-scaled`.
        Default is True.
    
    Returns
    --------
    pandas.DataFrame
        A copy of the DataFrame with the scaled column 
        replacing the original column if `inplace` is set to `True`.
        If `inplace` is set to `False`, 
        the copy of DataFrame is returned with the new scaled column added, 
        keeping the original column.

    Raises
    ------
    TypeError
        If the input `data` is not a pandas DataFrame.
    KeyError:
        If the column passed for scaling does not exist in the DataFrame.
    ValueError:
        If the column passed for scaling is not numeric.
        If the `method` is not `minmax` or `standard`.
        If the `new_min` value is greater or equal to the `new_max` when using `minmax` method.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"price": [25, 50, 75]})
    >>> df_scaled = column_scaler(df, column = 'price', method='minmax', new_min=0, new_max=1)
    >>> print(df_scaled)
            price
            0.0
            0.5
            1.0
    """
    # Check input is pd.DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    # Empty df warning
    if data.empty:
        warnings.warn(
            "Empty DataFrame detected. Empty DataFrame will be returned.", 
            UserWarning
            )
        return data.copy()
    
    # Error handling
    if column not in data.columns:
        raise KeyError("Column not found in the DataFrame.")
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError("Column must have numeric values.")
    if method not in {"minmax", "standard"}:
        raise ValueError("Invalid method. Method should be `minmax` or `standard`.")
    
    # Edge case warning
    if data[column].isna().any():
        warnings.warn(
            f"NaN value detected in column '{column}'. They will be unchanged", 
            UserWarning
            )

    if data[column].nunique() == 1:
        if method == "minmax":
            warnings.warn(
                "Single-value column detected. "
                "All values will be scaled to the midpoint of the `new_min` and `new_max`.",
                  UserWarning
                  )
            midpoint = (new_min + new_max) / 2
            scaled_column = pd.Series([midpoint] * len(data), index=data.index)
        
        elif method == "standard":
            warnings.warn(
                "Standard deviation is zero. "
                "All values are set to 0 to prevent division by zero.",
                UserWarning
                ) 
            scaled_column = pd.Series([0] * len(data), index=data.index) 

    # Scale the column
    else:
        # minmax scaling
        if method == "minmax":
            if new_min >= new_max:
                raise ValueError("`new_min` cannot be greater than `new_max`.")
            min_value = data[column].min()
            max_value = data[column].max()
            scaled_column = (
                ((data[column] - min_value) / (max_value - min_value)) 
            * (new_max - new_min) 
            + new_min
            )
            
        # standard scaling
        elif method == "standard":
            mean_value = data[column].mean() 
            std_value = data[column].std()
            scaled_column = (data[column] - mean_value) / std_value
    
    # Return df with scaled column
    if inplace:
        data[column] = scaled_column
        return data
    else:
        scaled_column_name = f"{column}_scaled"
        data[scaled_column_name] = scaled_column
        return data