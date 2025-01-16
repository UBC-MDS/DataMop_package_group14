import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datamop import datamop


# Fixtures for test data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
    'a': [10, np.nan, 30],
    'b': [1.5, 2.5, np.nan],
    'c': ['x', np.nan, 'z']
    })

@pytest.fixture
def single_row_data():
    return pd.DataFrame({'a': [np.nan], 'b': [5]})

@pytest.fixture
def single_column_data():
    return pd.DataFrame({'a': [1.0, np.nan, 3.0]})

@pytest.fixture
def empty_data():
    return pd.DataFrame()

@pytest.fixture
def all_missing_column():
    return pd.DataFrame({'a': [np.nan, np.nan, np.nan]})

@pytest.fixture
def no_missing_data():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

@pytest.fixture
def duplicates_data():
    return pd.DataFrame({'a': ['x', 'x', 'y', 'y', np.nan]})

# ** Expected Use Cases **

# Expected Use Cases 1: different strategies
def test_mean_strategy(sample_data):
    """
    Tests the 'mean' strategy without specifying columns.
    Numeric columns are filled with their mean, 
    and non-numeric columns remain unchanged.
    """
    result = datamop.sweep_nulls(sample_data, strategy='mean')
    expected = pd.DataFrame({
        'a': [10.0, 20.0, 30.0],
        'b': [1.5, 2.5, 2.0],
        'c': ['x', np.nan, 'z']
    })
    assert result['a'].isnull().sum() == 0 
    assert result['b'].isnull().sum() == 0 
    assert result['c'].isnull().sum() == 1 
    assert result.equals(expected)

def test_median_strategy(sample_data):
    """
    Tests the 'median' strategy without specifying columns.
    Numeric columns are filled with their median, 
    and non-numeric columns remain unchanged.
    """
    result = datamop.sweep_nulls(sample_data, strategy='median')
    expected = pd.DataFrame({
        'a': [10.0, 20.0, 30.0],
        'b': [1.5, 2.5, 2.0],
        'c': ['x', np.nan, 'z']
    })
    assert result['a'].isnull().sum() == 0 
    assert result['b'].isnull().sum() == 0 
    assert result['c'].isnull().sum() == 1 
    assert result.equals(expected)

def test_mode_strategy(sample_data):
    """
    Tests the 'mode' strategy without specifying columns.
    Columns are filled with their mode.
    """
    result = datamop.sweep_nulls(sample_data, strategy='mode')
    expected = pd.DataFrame({
        'a': [10.0, 10.0, 30.0],
        'b': [1.5, 2.5, 1.5],
        'c': ['x', 'x', 'z']
    })
    assert result['a'].isnull().sum() == 0 
    assert result['b'].isnull().sum() == 0 
    assert result['c'].isnull().sum() == 0 
    assert result.equals(expected)

def test_constant_strategy(sample_data):
    """
    Tests the 'constant' strategy without specifying columns.
    All missing values are replaced with the provided constant value.
    """
    result = datamop.sweep_nulls(sample_data, strategy='constant', fill_value=0)
    expected = pd.DataFrame({
        'a': [10.0, 0, 30.0],
        'b': [1.5, 2.5, 0],
        'c': ['x', 0, 'z']
    })
    assert result.equals(expected)

def test_drop_strategy_no_columns(sample_data):
    """
    Tests the 'drop' strategy without specifying columns.
    Rows with missing values in any column are dropped.
    """
    result = datamop.sweep_nulls(sample_data, strategy='drop')
    expected = sample_data.dropna()
    assert result.equals(expected)

def test_drop_strategy_with_columns(sample_data):
    """
    Tests the 'drop' strategy with specific columns.
    Rows with missing values in those specific column are dropped.
    """
    result = datamop.sweep_nulls(sample_data, strategy='drop', columns=['a', 'b'])
    expected = sample_data.dropna(subset=['a', 'b'])
    assert result.equals(expected)

# Expected Use Cases 2: apply to specific columns
def test_specific_columns(sample_data):
    """
    Tests applying the 'mean' strategy to specific columns.
    Only the specified columns are filled with their mean.
    """
    result = datamop.sweep_nulls(sample_data, strategy='mean', columns=['a'])
    expected = pd.DataFrame({
        'a': [10.0, 20.0, 30.0], 
        'b': [1.5, 2.5, np.nan], 
        'c': ['x', np.nan, 'z']
    })
    assert result['a'].isnull().sum() == 0 
    assert result['b'].isnull().sum() == 1
    assert result['c'].isnull().sum() == 1
    assert result.equals(expected)

# ** Edge Cases **

# Edge Cases 1: empty dataframe
def test_empty_dataframe(empty_data):
    """
    Tests an empty DataFrame.
    It is returned unchanged regardless of the strategy.
    """
    result = datamop.sweep_nulls(empty_data, strategy='mean')
    expected = pd.DataFrame()
    assert result.equals(expected)

# Edge Cases 2: dataframe with no missing values
def test_no_missing_values(no_missing_data): 
    """
    Tests a DataFrame with no missing values.
    It is returned unchanged.
    """
    result = datamop.sweep_nulls(no_missing_data, strategy='mean')
    expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert result.equals(expected)

# Edge Cases 3: dataframe with only one row
def test_single_row_data(single_row_data):
    """
    Tests a single-row DataFrame.
    Columns with all missing values are dropped with an warning, 
    other columns are handled by specified strategy correctly.
    """
    with pytest.warns(UserWarning, match="Column 'a' contains only missing values. Dropping the column."):
        result_mean = datamop.sweep_nulls(single_row_data, strategy='mean')
    expected_mean = pd.DataFrame({'b': [5]})
    assert result_mean.equals(expected_mean)

# Edge Cases 4: dataframe with only one column
def test_single_column_data(single_column_data):
    """
    Tests a single-column DataFrame.
    Missing values are handled by specified strategy correctly.
    """
    result_mean = datamop.sweep_nulls(single_column_data, strategy='mean')
    expected_mean = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
    assert result_mean.equals(expected_mean)

# Edge Cases 5: duplicate modes
def test_duplicate_modes(duplicates_data):
    """
    Tests a DataFrame with duplicate modes.
    The first mode is used to fill missing values.
    """
    result = datamop.sweep_nulls(duplicates_data, strategy='mode')
    expected = pd.DataFrame({'a': ['x', 'x', 'y', 'y', 'x']})
    assert result.equals(expected)

# Edge Cases 6: column with no values (all missing)
def test_all_missing_column(all_missing_column):
    """
    Tests a column with all missing values.
    It is dropped with a warning.
    """
    with pytest.warns(UserWarning, match="Column 'a' contains only missing values. Dropping the column."):
        result = datamop.sweep_nulls(all_missing_column, strategy='mean')
    expected = pd.DataFrame(index=all_missing_column.index)
    assert result.equals(expected)

# Edge Cases 7: empty columns list
def test_empty_columns_list(sample_data):
    """
    Tests an empty column list.
    The strategy is applied to all columns with a warning.
    """
    with pytest.warns(UserWarning, match="Columns list is empty. Applying strategy to all columns."):
        result = datamop.sweep_nulls(sample_data, strategy='mean', columns=[])
        expected = pd.DataFrame({
        'a': [10.0, 20.0, 30.0],
        'b': [1.5, 2.5, 2.0],
        'c': ['x', np.nan, 'z']
    })
    assert result.equals(expected)

# Edge Cases 8: Non-numeric columns with numeric strategies
def test_non_numeric_with_numeric_strategy(sample_data):
    """
    Tests applying a numeric strategy to non-numeric columns
    A warning is raised and no changes are made to the column.
    """
    with pytest.warns(UserWarning, match="Strategy 'mean' cannot be applied to non-numeric column 'c'"):
        datamop.sweep_nulls(sample_data, strategy='mean', columns=['c'])

# ** Error Cases **

# Error Case 1: Unsupported data type
def test_unsupported_data_type():
    """
    Tests unsupported data types.
    A ValueError is raised for inputs that are not pandas DataFrames.
    """
    with pytest.raises(ValueError, match="Input data must be a pandas DataFrame"):
        datamop.sweep_nulls("not a dataframe", strategy='mean')
        datamop.sweep_nulls(pd.Series([1, 2, np.nan]), strategy='mean')

# Error Case 2: Invalid strategy
def test_invalid_strategy(sample_data):
    """
    Tests an invalid strategy.
    A ValueError is raised for unsupported strategy.
    """
    with pytest.raises(ValueError, match="Unsupported strategy. Choose from 'mean', 'median', 'mode', 'constant', or 'drop'"):
        datamop.sweep_nulls(sample_data, strategy='invalid')

# Error Case 3: Invalid column name
def test_invalid_column_name(sample_data):
    """
    Tests an invalid column name.
    A KeyError is raised if a specified column does not exist in the DataFrame.
    """
    with pytest.raises(KeyError, match="Column 'nonexistent_column' not found in the DataFrame."):
        datamop.sweep_nulls(sample_data, strategy='mean', columns=['nonexistent_column'])

# Error Case 4: Missing `fill_value` for constant strategy
def test_missing_fill_value(sample_data):
    """
    Tests a missing `fill_value` for the 'constant' strategy.
    A ValueError is raised when missing `fill_value` for the 'constant' strategy.
    """
    with pytest.raises(ValueError, match="`fill_value` must be provided for 'constant' strategy."):
        datamop.sweep_nulls(sample_data, strategy='constant')

# Error Case 5: Invalid `fill_value` type
def test_invalid_fill_value(sample_data):
    """
    Tests an invalid `fill_value` type.
    A TypeError is raised when the type is not supported.
    """
    with pytest.raises(TypeError, match="Invalid `fill_value` type"):
        datamop.sweep_nulls(sample_data, strategy='constant', fill_value={"key": "value"})