import pandas as pd

def column_encoder(df, columns, method='one-hot', order=None):
    """
    Encodes categorical columns using one-hot or ordinal encoding based on user input.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the dataset.
    column : str
        The name of the column to be encoded. This column must be categorical.
    method : str, optional, default='one-hot'
        The encoding method to use. Accepts either 'one-hot' for one-hot encoding
        or 'ordinal' for ordinal encoding.
    order : dict, optional, default=None
        A dictionary specifying the custom order for ordinal encoding.
        The keys should be column names, and the values should be lists
        defining the order of categories for each column.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the specified column encoded. The original column 
        will be dropped.

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
    ...     'Level': ['A', 'B', 'C', 'D']
    ... })

    >>> encoded_df_onehot = column_encoder(df, columns=['Sport'], method='one-hot')
    >>> print(encoded_df_onehot)
      Level  Sport_Badminton  Sport_Basketball  Sport_Football  Sport_Tennis
         A                0                 0               0             1
         B                0                 1               0             0
         C                0                 0               1             0
         D                1                 0               0             0
         
    >>> encoded_df_ordinal = column_encoder(df, columns=['Level'], method='ordinal', order={'Level': ['A', 'B', 'C', 'D']})
    >>> print(encoded_df_ordinal)
            Sport  Level
           Tennis      0
        Basketball     1
         Football      2
        Badminton      3

    """
    encoded_df = df
    if method == 'one-hot':
        for column in columns:
            dummies = pd.get_dummies(encoded_df[column], prefix=column)
            encoded_df = pd.concat([encoded_df.drop(column, axis=1), dummies], axis=1)
    elif method == 'ordinal':
        if order is None:
            raise ValueError("You must provide an 'order' parameter for ordinal encoding.")
        
        for column in columns:
            if column not in order:
                raise ValueError(f"Order for column '{column}' is not provided.")
            
            custom_order = order[column]
            if not set(encoded_df[column].unique()).issubset(set(custom_order)):
                raise ValueError(f"Order for column '{column}' does not match its unique values.")
            
            val_to_int = {val: idx for idx, val in enumerate(custom_order)}
            encoded_df[column] = encoded_df[column].map(val_to_int)
            
    return encoded_df