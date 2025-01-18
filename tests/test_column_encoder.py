import pytest
import pandas as pd
from datamop.column_encoder import column_encoder

@pytest.fixture
def level_df():
    """DataFrame with two columns: 'Sport' and 'Level'."""
    return pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })

@pytest.fixture
def level_complexity_df():
    """DataFrame with three columns: 'Sport', 'Level', 'Complexity'."""
    return pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D'],
        'Complexity': ['Hard', 'Easy', 'Mild', 'Easy']
    })

@pytest.fixture
def single_value_level_df():
    """DataFrame where 'Level' has a single repeated value."""
    return pd.DataFrame({
        'Level': ['A', 'A', 'A', 'A']
    })

@pytest.fixture
def empty_df():
    """Empty DataFrame."""
    return pd.DataFrame()

@pytest.fixture
def missing_values_df():
    """DataFrame with a missing value in 'Sport'."""
    return pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', None, 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })

@pytest.fixture
def sport_only_df():
    """DataFrame with a single column 'Sport'."""
    return pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton']
    })


# Expected cases:
def test_case_1_one_hot_single_column(level_df):
    """User input dataframe, set method equal one hot, leave order empty, input one column, output correct dataframe after one hot encoded"""
    result_df_one = column_encoder(level_df, columns=['Sport'], method='one-hot')
    expected_columns_one = ['Level', 'Sport_Badminton', 'Sport_Basketball', 'Sport_Football', 'Sport_Tennis']
    
    assert list(result_df_one.columns) == expected_columns_one
    assert result_df_one.shape == (4, 5)
    assert result_df_one['Sport_Tennis'].tolist() == [1, 0, 0, 0]
    assert result_df_one['Sport_Badminton'].tolist() == [0, 0, 0, 1]

def test_case_2_one_hot_multiple_columns(level_complexity_df):
    """User input dataframe, set method equal one hot, leave order empty, input multiple columns, output correct dataframe after one hot encoded"""
    result_df_one2 = column_encoder(level_complexity_df, columns=['Sport', 'Complexity'], method='one-hot')
    expected_columns_one2 = ['Level', 'Sport_Badminton', 'Sport_Basketball', 'Sport_Football', 'Sport_Tennis', 'Complexity_Easy', 'Complexity_Hard', 'Complexity_Mild']
    
    assert result_df_one2.shape == (4, 8)
    assert list(result_df_one2.columns) == expected_columns_one2
    assert result_df_one2['Sport_Tennis'].tolist() == [1, 0, 0, 0]
    assert result_df_one2['Complexity_Hard'].tolist() == [1, 0, 0, 0]

def test_case_3_ordinal_single_column(level_df):
    """User input all parameters, set method equal ordinal, input correct order, output correct dataframe after ordinal encoded"""
    custom_order = {'Level': ['A', 'B', 'C', 'D']}
    result_df_ord = column_encoder(level_df, columns=['Level'], method='ordinal', order=custom_order)
    
    expected_columns_ord = ['Sport', 'Level']
    assert list(result_df_ord.columns) == expected_columns_ord
    assert result_df_ord['Level'].tolist() == [0, 1, 2, 3]

def test_case_4_ordinal_multiple_columns(level_complexity_df):
    """User input dataframe, set method equal ordinal, input correct order, input multiple columns, output correct dataframe after ordinal encoded"""
    custom_order = {
        'Level': ['A', 'B', 'C', 'D'],
        'Complexity': ['Easy', 'Mild', 'Hard']
    }
    
    result_df_ord_multi = column_encoder(level_complexity_df, columns=['Level', 'Complexity'], method='ordinal', order=custom_order)
    
    expected_columns_ord_multi = ['Sport', 'Level', 'Complexity']
    assert list(result_df_ord_multi.columns) == expected_columns_ord_multi
    assert result_df_ord_multi['Level'].tolist() == [0, 1, 2, 3]
    assert result_df_ord_multi['Complexity'].tolist() == [2, 0, 1, 0]

# Edge cases:
def test_case_5_single_value_in_order_column(single_value_level_df):
    """User input dataframe containing only one value for the required input order column, should raise a warning"""
    custom_order = {'Level': ['A']}
    with pytest.warns(UserWarning, match="The column 'Level' contains only one unique value"):
        column_encoder(single_value_level_df, columns=['Level'], method='ordinal', order=custom_order)

def test_case_6_empty_dataframe(empty_df):
    """User input an empty dataframe, should output an empty dataframe"""
    result_df = column_encoder(empty_df, columns=[], method='one-hot')
    assert result_df.empty  # Check if the output dataframe is empty

def test_case_7_missing_values_handling(missing_values_df):
    """If missing value is in the dataframe, should raise a warning and leave as null value."""
    custom_order = {'Level': ['A', 'B', 'C', 'D']}
    with pytest.warns(UserWarning, match="Missing values detected. They will be left as null"):
        result_df = column_encoder(missing_values_df, columns=['Level'], method='ordinal', order=custom_order)
    
    assert result_df.isnull().any().any()

# Error cases:
def test_case_8_one_hot_with_order_error(level_df):
    """User input dataframe, set method equal one hot, input order, the output should raise a value error"""
    custom_order = {'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton']}
    
    with pytest.raises(ValueError, match="Order parameter is not applicable for method 'one-hot'"):
 
        column_encoder(level_df, columns=['Sport'], method='one-hot', order=custom_order)

def test_case_9_ordinal_incomplete_order_error(level_df):
    """User input all parameters, set method equal ordinal, input order, but order does not contain all values for the column, the output should raise a value error"""
    incomplete_order = {'Level': ['A', 'B']}   
    with pytest.raises(ValueError, match="Order for column 'Level' does not match its unique values"):
        column_encoder(level_df, columns=['Level'], method='ordinal', order=incomplete_order)

def test_case_10_order_column_not_in_column_list(level_df):
    """User input dataframe, set method equal ordinal, input order, but the order column is not in the column list, the output should raise a value error"""
    custom_order = {
        'Complexity': ['Easy', 'Medium', 'Hard'],
        'Level': ['A', 'B', 'C', 'D']
        }
    with pytest.raises(ValueError, match="The column 'Complexity' specified in order is not in the column list"):
        column_encoder(level_df, columns=['Level'], method='ordinal', order=custom_order)

def test_case_11_missing_order_for_ordinal(level_df):
    """ User input dataframe, set method equal ordinal, does not input order, the output should raise a value error ask user to input an order"""
    with pytest.raises(ValueError, match="Order must be specified for ordinal encoding"):
        column_encoder(level_df, columns=['Level'], method='ordinal')

def test_case_12_missing_required_parameter(sport_only_df):
    """Missing required parameter, should output an error"""
    with pytest.raises(TypeError, match="Columns parameter must be a list of strings"):
        column_encoder(sport_only_df, columns=None, method='one-hot')

    with pytest.raises(TypeError, match="Method parameter must be a string"):
        column_encoder(sport_only_df, columns=['Sport'], method=None)


def test_case_13_invalid_method(sport_only_df):
    """User input dataframe, but the method used is not ordinal nor onehot, should raise an error."""
    with pytest.raises(ValueError, match="Invalid method specified. Use 'one-hot' or 'ordinal'"):
        column_encoder(sport_only_df, columns=['Sport'], method='binary')  # Invalid method

def test_case_14_column_not_in_dataframe(sport_only_df):
    """User input dataframe, input column, but the column is not in the dataframe, should output an error"""
    with pytest.raises(KeyError, match="The column 'Level' is not in the dataframe"):
        column_encoder(sport_only_df, columns=['Level'], method='ordinal', order={'Level': ['A', 'B', 'C', 'D']})

def test_case_15_incorrect_input_type():
    """User inputs incorrect data type (non-DataFrame), should raise a TypeError"""
    incorrect_input = [{'Sport': 'Tennis', 'Level': 'A'}, {'Sport': 'Basketball', 'Level': 'B'}]
    
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        column_encoder(incorrect_input, columns=['Sport'], method='one-hot')

def test_case_16_incorrect_columns_type(level_df):
    """User inputs incorrect data type for columns parameter, should raise a TypeError"""
    with pytest.raises(TypeError, match="Columns parameter must be a list of strings"):
        column_encoder(level_df, columns='Sport', method='one-hot')
    with pytest.raises(TypeError, match="Order parameter must be a dictionary"):
        column_encoder(level_df, columns=['Level'], method='ordinal', order=[('Level', ['A', 'B', 'C', 'D'])])
    with pytest.raises(TypeError, match="Method parameter must be a string"):
        column_encoder(level_df, columns=['Sport'], method=['one-hot'])
