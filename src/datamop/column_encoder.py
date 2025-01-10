

def column_encoder(df, columns, method='one-hot'):
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
         
    >>> encoded_df_ordinal = column_encoder(df, columns=['Level'], method='ordinal')
    >>> print(encoded_df_ordinal)
            Sport  Level
           Tennis      0
        Basketball     1
         Football      2
        Badminton      3

    """
    
    return None