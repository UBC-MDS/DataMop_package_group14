import pytest
import pandas as pd
from datamop.column_encoder import column_encoder

# Expected cases:
# test case 1: User input dataframe, set method equal one hot, leave order empty, input one column, output correct dataframe after one hot encoded
def test_case_1_one_hot_single_column():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    
    result_df_one = column_encoder(df, columns=['Sport'], method='one-hot')
    expected_columns_one = ['Level', 'Sport_Badminton', 'Sport_Basketball', 'Sport_Football', 'Sport_Tennis']
    
    assert list(result_df_one.columns) == expected_columns_one
    assert result_df_one.shape == (4, 5)
    assert result_df_one['Sport_Tennis'].tolist() == [1, 0, 0, 0]
    assert result_df_one['Sport_Badminton'].tolist() == [0, 0, 0, 1]

# test case 2: User input dataframe, set method equal one hot, leave order empty, input multiple columns, output correct dataframe after one hot encoded
def test_case_2_one_hot_multiple_columns():
    df_mul = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D'],
        'Complexity': ['Hard', 'Easy', 'Mild', 'Easy']
    })
    
    result_df_one2 = column_encoder(df_mul, columns=['Sport', 'Complexity'], method='one-hot')
    expected_columns_one2 = ['Level', 'Sport_Badminton', 'Sport_Basketball', 'Sport_Football', 'Sport_Tennis', 'Complexity_Easy', 'Complexity_Hard', 'Complexity_Mild']
    
    assert result_df_one2.shape == (4, 8)
    assert list(result_df_one2.columns) == expected_columns_one2
    assert result_df_one2['Sport_Tennis'].tolist() == [1, 0, 0, 0]
    assert result_df_one2['Complexity_Hard'].tolist() == [1, 0, 0, 0]

# test case 3: User input all parameters, set method equal ordinal, input correct order, output correct dataframe after ordinal encoded
def test_case_3_ordinal_single_column():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    
    custom_order = {'Level': ['A', 'B', 'C', 'D']}
    result_df_ord = column_encoder(df, columns=['Level'], method='ordinal', order=custom_order)
    
    expected_columns_ord = ['Sport', 'Level']
    assert list(result_df_ord.columns) == expected_columns_ord
    assert result_df_ord['Level'].tolist() == [0, 1, 2, 3]

# test case 4: User input dataframe, set method equal ordinal, input correct order, input multiple columns, output correct dataframe after ordinal encoded
def test_case_4_ordinal_multiple_columns():
    df_mul = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D'],
        'Complexity': ['Hard', 'Easy', 'Mild', 'Easy']
    })
    
    custom_order = {
        'Level': ['A', 'B', 'C', 'D'],
        'Complexity': ['Easy', 'Mild', 'Hard']
    }
    
    result_df_ord_multi = column_encoder(df_mul, columns=['Level', 'Complexity'], method='ordinal', order=custom_order)
    
    expected_columns_ord_multi = ['Sport', 'Level', 'Complexity']
    assert list(result_df_ord_multi.columns) == expected_columns_ord_multi
    assert result_df_ord_multi['Level'].tolist() == [0, 1, 2, 3]
    assert result_df_ord_multi['Complexity'].tolist() == [2, 0, 1, 0]

# Edge cases:
# test case 5: User input dataframe containing only one value for the required input order column, should raise a warning
def test_case_5_single_value_in_order_column():
    df = pd.DataFrame({
        'Level': ['A', 'A', 'A', 'A']
    })
    
    custom_order = {'Level': ['A']}
    with pytest.warns(UserWarning, match="The column 'Level' contains only one unique value"):
        column_encoder(df, columns=['Level'], method='ordinal', order=custom_order)

# test case 6: User input an empty dataframe, should output an empty dataframe
def test_case_6_empty_dataframe():
    df_empty = pd.DataFrame()
    
    result_df = column_encoder(df_empty, columns=[], method='one-hot')
    assert result_df.empty  # Check if the output dataframe is empty

# test case 7: If missing value is in the dataframe, should raise a warning and leave as null value
def test_case_7_missing_values_handling():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', None, 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    
    custom_order = {'Level': ['A', 'B', 'C', 'D']}
    with pytest.warns(UserWarning, match="Missing values detected. They will be left as null"):
        result_df = column_encoder(df, columns=['Level'], method='ordinal', order=custom_order)
    
    assert result_df.isnull().any().any()

# Error cases:
# test case 8: User input dataframe, set method equal one hot, input order, the output should raise a value error
def test_case_8_one_hot_with_order_error():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    
    with pytest.raises(ValueError, match="Order parameter is not applicable for method 'one-hot'"):
        custom_order = {'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton']}
        column_encoder(df, columns=['Sport'], method='one-hot', order=custom_order)

# test case 9: User input all parameters, set method equal ordinal, input order, but order does not contain all values for the column, the output should raise a value error
def test_case_9_ordinal_incomplete_order_error():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    
    with pytest.raises(ValueError, match="Order for column 'Level' does not match its unique values"):
        incomplete_order = {'Level': ['A', 'B']}
        column_encoder(df, columns=['Level'], method='ordinal', order=incomplete_order)

# test case 10: User input dataframe, set method equal ordinal, input order, but the order column is not in the column list, the output should raise a value error
def test_case_10_order_column_not_in_column_list():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    
    custom_order = {
        'Complexity': ['Easy', 'Medium', 'Hard'],
        'Level': ['A', 'B', 'C', 'D']
        }
    with pytest.raises(ValueError, match="The column 'Complexity' specified in order is not in the column list"):
        column_encoder(df, columns=['Level'], method='ordinal', order=custom_order)

# test case 11: User input dataframe, set method equal ordinal, does not input order, the output should raise a value error ask user to input an order
def test_case_11_missing_order_for_ordinal():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    
    with pytest.raises(ValueError, match="Order must be specified for ordinal encoding"):
        column_encoder(df, columns=['Level'], method='ordinal')

# test case 12: Missing required parameter, should output an error
def test_case_12_missing_required_parameter():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton']
    })

    # Check for missing columns parameter
    with pytest.raises(TypeError, match="Columns parameter must be a list of strings"):
        column_encoder(df, columns=None, method='one-hot')

    # Check for missing method parameter
    with pytest.raises(TypeError, match="Method parameter must be a string"):
        column_encoder(df, columns=['Sport'], method=None)


# test case 13: User input dataframe, but the method used is not ordinal nor onehot, should raise an error.
def test_case_13_invalid_method():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton']
    })
    
    with pytest.raises(ValueError, match="Invalid method specified. Use 'one-hot' or 'ordinal'"):
        column_encoder(df, columns=['Sport'], method='binary')  # Invalid method

# test case 14: User input dataframe, input column, but the column is not in the dataframe, should output an error
def test_case_14_column_not_in_dataframe():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton']
    })
    
    with pytest.raises(KeyError, match="The column 'Level' is not in the dataframe"):
        column_encoder(df, columns=['Level'], method='ordinal', order={'Level': ['A', 'B', 'C', 'D']})

# test case 15: User inputs incorrect data type (non-DataFrame), should raise a TypeError
def test_case_15_incorrect_input_type():
    incorrect_input = [{'Sport': 'Tennis', 'Level': 'A'}, {'Sport': 'Basketball', 'Level': 'B'}]
    
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        column_encoder(incorrect_input, columns=['Sport'], method='one-hot')

# test case 16: User inputs incorrect data type for columns parameter, should raise a TypeError
def test_case_16_incorrect_columns_type():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    
    with pytest.raises(TypeError, match="Columns parameter must be a list of strings"):
        column_encoder(df, columns='Sport', method='one-hot')
    with pytest.raises(TypeError, match="Order parameter must be a dictionary"):
        column_encoder(df, columns=['Level'], method='ordinal', order=[('Level', ['A', 'B', 'C', 'D'])])
    with pytest.raises(TypeError, match="Method parameter must be a string"):
        column_encoder(df, columns=['Sport'], method=['one-hot'])
