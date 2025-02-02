import pandas as pd


def column_encoder(data, columns, method='one-hot', order=None):
    """
    Encodes categorical columns using one-hot or ordinal encoding based on user input.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing the dataset.
    columns : list
        The name of the columns to be encoded. 
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
    
    Raises:
    -------
    TypeError:
        If input types are incorrect (e.g., non-DataFrame input, columns not a list of strings,
        method not a string, or order not a dictionary).
    ValueError:
        If required parameters are missing or invalid values are provided.
    KeyError:
        If specified columns are not found in the input DataFrame.
    UserWarning:
        If a column contains only one unique value or if there are missing values.

    Examples:
    ---------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
    ...     'Level': ['A', 'B', 'C', 'D']
    ... })

    >>> encoded_df_onehot = column_encoder(data, columns=['Sport'], method='one-hot')
    >>> print(encoded_df_onehot)
      Level  Sport_Badminton  Sport_Basketball  Sport_Football  Sport_Tennis
         A                0                 0               0             1
         B                0                 1               0             0
         C                0                 0               1             0
         D                1                 0               0             0
         
    >>> encoded_df_ordinal = column_encoder(data, columns=['Level'], method='ordinal', order={'Level': ['A', 'B', 'C', 'D']})
    >>> print(encoded_df_ordinal)
            Sport  Level
           Tennis      0
        Basketball     1
         Football      2
        Badminton      3

    """
    #check input type
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise TypeError("Columns parameter must be a list of strings")
    if not isinstance(method, str):
        raise TypeError("Method parameter must be a string")
    if method == 'ordinal' and order is not None and not isinstance(order, dict):
        raise TypeError("Order parameter must be a dictionary")
    encoded_df = data.copy()
    
    if method == 'one-hot':
        #check if order is input
        if order is not None:
            raise ValueError("Order parameter is not applicable for method 'one-hot'")
        
        for column in columns:
            #check if column is in dataframe
            if column not in encoded_df.columns:
                raise KeyError(f"The column '{column}' is not in the dataframe")
            dummies = pd.get_dummies(encoded_df[column], prefix=column).astype(int)
            encoded_df = pd.concat([encoded_df.drop(column, axis=1), dummies], axis=1)
            
    elif method == 'ordinal':
        #check if order is input
        if order is None:
            raise ValueError("Order must be specified for ordinal encoding")
        
        for column in columns:
            #check if column is in dataframe
            if column not in encoded_df.columns:
                raise KeyError(f"The column '{column}' is not in the dataframe")
            #check if order is in the input column
            if column not in order:
                raise ValueError(f"Order for column '{column}' is not provided")
            
        for column in order:
            #check if column is in the order
            if column not in columns:
                raise ValueError(f"The column '{column}' specified in order is not in the column list")

            custom_order = order[column]
            unique_values = encoded_df[column].unique()
            #check if order match what is inside column
            if not set(unique_values).issubset(set(custom_order)):
                raise ValueError(f"Order for column '{column}' does not match its unique values")
            
            if len(unique_values) == 1:
                import warnings
                warnings.warn(f"The column '{column}' contains only one unique value", UserWarning)
            
            val_to_int = {val: idx for idx, val in enumerate(custom_order)}
            encoded_df[column] = encoded_df[column].map(val_to_int)
    else:
        raise ValueError("Invalid method specified. Use 'one-hot' or 'ordinal'")
    
    if encoded_df.isnull().any().any():
        import warnings
        warnings.warn("Missing values detected. They will be left as null", UserWarning)
            
    return encoded_df