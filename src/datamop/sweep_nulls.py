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