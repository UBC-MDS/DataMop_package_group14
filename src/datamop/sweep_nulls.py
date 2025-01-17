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

    strategy : {'mean', 'median', 'mode', 'constant', 'drop'}, optional, default='mean'
        The strategy to use for handling missing values. Supported options are:
        - 'mean': For numeric columns only. Replace missing values with the mean of the respective column.
        - 'median': For numeric columns only. Replace missing values with the median of the respective column.
        - 'mode': Replace missing values with the mode (most frequent value) of the respective column.
        - 'constant': Replace missing values with a specified constant value (requires `fill_value`).
        - 'drop': Drop rows or columns containing missing values (depending on the `columns` parameter).

    columns : list of str or None, optional, default=None
        The specific columns to apply the missing value handling. 
        If None or an empty list, the strategy is applied to all columns.

    fill_value : int, float, str, or None, optional, default=None
        The constant value to use when `strategy='constant'`. Ignored for other strategies.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with missing values handled based on the specified strategy.

    Raises
    ------
    ValueError
        - If the input data is not a pandas.DataFrame.
        - If the input strategy is not in 'mean', 'median', 'mode', 'constant', or 'drop'.
        - If `fill_value` is missing for the 'constant' strategy.
    KeyError
        If any specified column in `columns` does not exist in the pandas.DataFrame.
    TypeError
        If the input of `fill_value` is not a number or a string.


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

    # Ensure the input data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    # If no columns are specified (default or empty list), apply strategy to all columns
    if not columns:
        warnings.warn("Columns list is empty. Applying strategy to all columns.", UserWarning)
        columns = data.columns

    # Check if the provided strategy is valid
    if strategy not in ['mean', 'median', 'mode', 'constant', 'drop']:
        raise ValueError("Unsupported strategy. Choose from 'mean', 'median', 'mode', 'constant', or 'drop'")
    
    # `fill_value` is required for the 'constant' strategy
    if strategy == 'constant' and fill_value is None:
        raise ValueError("`fill_value` must be provided for 'constant' strategy.")
    
    # Store the original data types
    original_dtypes = data.dtypes.to_dict()

    # Loop through each column and handle missings
    for column in columns:

        # Raise error if the column is not found in the DataFrame
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in the DataFrame.")
        
        # Drop column if the entire is missing
        if data[column].isna().all(): 
            warnings.warn(f"Column '{column}' contains only missing values. Dropping the column.", UserWarning)
            data = data.drop(columns=[column])
            continue

        # Numeric columns
        if data[column].dtype in ['int64', 'float64']: 
            if strategy == 'mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 'median':
                data[column] = data[column].fillna(data[column].median())
            elif strategy == 'mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'constant':
                if not isinstance(fill_value, (int, float)):
                    raise TypeError("Invalid `fill_value` type.")
                data[column] = data[column].fillna(fill_value)
            elif strategy == 'drop':
                data = data.dropna(subset=[column])

        # Non-numeric columns
        else: 
            if strategy in ['mean', 'median']:
                warnings.warn(f"Strategy '{strategy}' cannot be applied to non-numeric column '{column}'", UserWarning)
                data[column] = data[column]        
            if strategy == 'mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'constant':
                if not isinstance(fill_value, (int, float, str)):
                    raise TypeError("Invalid `fill_value` type.")
                data[column] = data[column].fillna(fill_value)
            elif strategy == 'drop':
                data = data.dropna(subset=[column])

    # Restore the original data types
    remaining_columns = data.columns
    for column in remaining_columns:
        if column in original_dtypes:
            try:
                data[column] = data[column].astype(original_dtypes[column]) 
            except ValueError: 
                warnings.warn("Could not restore the original dtype for column '{column}'. Data type changes to {data[column].dtype}.", UserWarning)
    
    return data