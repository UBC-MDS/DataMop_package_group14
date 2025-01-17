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
        If `True` the column is replaced with new scaled column.
        If `False` the original column is retained and the new scaled column is added to the dataframe with title `<column-name>-scaled`.
        Default is True
    
    Returns
    --------
    pandas.DataFrame
        A copy of the DataFrame with the scaled column replacing the original column if `inplace` is set to True.
        If `inplace` is set to False, the copy of DataFrame is returned with the new scaled column added, keeping the original column.

    Raises
    ------
    ValueError:
        If the column passed for scaling is not numeric.
    KeyError:
        If the column passed for scaling does not exist in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"price": [25, 50, 75]})
    >>> df_scaled = column_scaler(df, column = 'price', method='minmax', new_min=0, new_max=1)
    >>> print(df_scaled)
            price
            0.25
            0.50
            0.75

    """
    pass
