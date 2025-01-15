import pandas as pd
import numpy as np
import warnings

def sweep_nulls(data, strategy='mean', columns=None, fill_value=None):
    """
    Handles missing values in a dataset using the specified strategy.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset where missing values need to be handled.

    strategy : str, optional, default='mean'
        The strategy to use for handling missing values. Supported options are:
        - 'mean': Replace missing values with the mean of the respective column.
        - 'median': Replace missing values with the median of the respective column.
        - 'mode': Replace missing values with the mode (most frequent value) of the respective column.
        - 'constant': Replace missing values with a specified constant value (requires `fill_value`).
        - 'drop': Drop rows or columns containing missing values (depending on the `columns` parameter).

    columns : list of str or None, optional, default=None
        The specific columns to apply the missing value handling. If None, the strategy is applied to all columns.

    fill_value : str/numeric, optional, default=None
        The constant value to use when `strategy='constant'`. Ignored for other strategies.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with missing values handled based on the specified strategy.

    Examples
    --------
        a    b     c
    0  10.0  1.5     x
    1   NaN  2.5  None
    2  30.0  NaN     z

    >>> cleaned = sweep_nulls(data, strategy='mean')
    >>> print(cleaned)
            a    b     c
        0  10.0  1.5     x
        1  20.0  2.5  None
        2  30.0  2.0     z
    
    """

    # handle all missings if column not specified
    if columns is None:
        columns = data.columns

    if strategy not in ['mean', 'median', 'mode', 'constant', 'drop']:
        raise ValueError("Unsupported strategy. Choose from 'mean', 'median', 'mode', 'constant', or 'drop'")
    
    # `fill_value` is required for the 'constant' strategy
    if strategy == 'constant' and fill_value is None:
        raise ValueError("`fill_value` must be provided for 'constant' strategy.")
    
    # Loop through each column in the dataframe
    for column in columns:
        original_dtype = data[column].dtype
        if original_dtype in ['int64', 'float64']: 
            if strategy == 'mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 'median':
                data[column] = data[column].fillna(data[column].median())
            elif strategy == 'mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'constant':
                data[column] = data[column].fillna(fill_value)
            elif strategy == 'drop':
                data = data.dropna(subset=[column])

            data[column] = data[column].astype(original_dtype)

        else: 
            if strategy in ['mean', 'median']:
                warnings.warn(f"Strategy '{strategy}' cannot be applied to non-numeric column '{column}'", UserWarning)
            if strategy == 'mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'constant':
                data[column] = data[column].fillna(fill_value)
            elif strategy == 'drop':
                data = data.dropna(subset=[column])
    
    # Drop columns with only missing values
    if strategy == 'drop':
        data = data.dropna(axis=1, how='all')
    
    return data

