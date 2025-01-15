import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datamop import sweep_nulls


# Fixtures for test data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
    'a': [10, None, 30],
    'b': [1.5, 2.5, None],
    'c': ['x', None, 'z']
    })

@pytest.fixture
def single_row_data():
    return pd.DataFrame({'a': [np.nan], 'b': [5]})

@pytest.fixture
def single_column_data():
    return pd.DataFrame({'a': [1, np.nan, 3]})

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
    result = sweep_nulls(sample_data, strategy='mean')
    expected = pd.DataFrame({
        'a': [10.0, 20.0, 30.0],
        'b': [1.5, 2.5, 2.0],
        'c': ['x', None, 'z']
    })
    assert result['a'].isnull().sum() == 0 
    assert result['b'].isnull().sum() == 0 
    assert result['c'].isnull().sum() == 1 
    assert result.equals(expected)

def test_median_strategy(sample_data):
    result = sweep_nulls(sample_data, strategy='median')
    expected = pd.DataFrame({
        'a': [10.0, 20.0, 30.0],
        'b': [1.5, 2.5, 2.0],
        'c': ['x', None, 'z']
    })
    assert result['a'].isnull().sum() == 0 
    assert result['b'].isnull().sum() == 0 
    assert result['c'].isnull().sum() == 1 
    assert result.equals(expected)

def test_mode_strategy(sample_data):
    result = sweep_nulls(sample_data, strategy='mode')
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
    result = sweep_nulls(sample_data, strategy='constant', fill_value=0)
    expected = pd.DataFrame({
        'a': [10.0, 0, 30.0],
        'b': [1.5, 2.5, 0],
        'c': ['x', 0, 'z']
    })
    assert result.equals(expected)

def test_drop_strategy_no_columns(sample_data):
    result = sweep_nulls(sample_data, strategy='drop')
    expected = sample_data.dropna()
    assert result.equals(expected)

def test_drop_strategy_with_columns(sample_data):
    result = sweep_nulls(sample_data, strategy='drop', columns=['a', 'b'])
    expected = sample_data.dropna(subset=['a', 'b'])
    assert result.equals(expected)

# Expected Use Cases 2: apply to specific columns
def test_specific_columns(sample_data):
    result = sweep_nulls(sample_data, strategy='mean', columns=['a'])
    expected = pd.DataFrame({
        'a': [10.0, 20.0, 30.0], 
        'b': [1.5, 2.5, None], 
        'c': ['x', None, 'z']
    })
    assert result['a'].isnull().sum() == 0 
    assert result['b'].isnull().sum() == 1
    assert result['c'].isnull().sum() == 1
    assert result.equals(expected)

# ** Edge Cases **

# Edge Cases 1: empty dataframe
def test_empty_dataframe(empty_dataframe):
    result = sweep_nulls(empty_dataframe, strategy='mean')
    expected = pd.DataFrame()
    assert result.equals(expected)

# Edge Cases 2: dataframe with no missing values
def test_no_missing_values(no_missing_data): 
    result = sweep_nulls(no_missing_data, strategy='mean')
    expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert result.equals(expected)

# Edge Cases 3: dataframe with only one row
def test_single_row_data(single_row_data):
    result_mean = sweep_nulls(single_column_data, strategy='mean', fill_value=0)
    expected_mean = pd.DataFrame({'a': np.nan, 'b': [5]})
    result_constant = sweep_nulls(single_row_data, strategy='constant', fill_value=0)
    expected_constant = pd.DataFrame({'a': [0], 'b': [5]})
    assert result_mean.equals(expected_mean)
    assert result_constant.equals(expected_constant)

# Edge Cases 4: dataframe with only one column
def test_single_column_data(single_column_data):
    result_mean = sweep_nulls(single_column_data, strategy='mean')
    expected_mean = pd.DataFrame({'a': [1, 2, 3]})
    result_constant = sweep_nulls(single_column_data, strategy='constant', fill_value=0)
    expected_constant = pd.DataFrame({'a': [1, 0, 3]})
    assert result_mean.equals(expected_mean)
    assert result_constant.equals(expected_constant)

# Edge Cases 5: duplicate modes
def test_duplicate_modes(duplicates_data):
    result = sweep_nulls(duplicates_data, strategy='mode')
    expected = pd.DataFrame({'a': ['x', 'x', 'y', 'y', 'x']})
    assert result.equals(expected)

# Edge Cases 6: column with no values (all missing)
def test_all_missing_column(all_missing_column):
    with pytest.warns(UserWarning, match="Column contains only missing values. Dropping the column."):
        result = sweep_nulls(all_missing_column, strategy='mean')
    expected = all_missing_column.dropna(axis=1)
    assert result.equals(expected)

# Edge Cases 7: empty columns list
def test_empty_columns_list(sample_data):
    with pytest.warns(UserWarning, match="Columns list is empty. Returning original DataFrame."):
        result = sweep_nulls(sample_data, strategy='mean', columns=[])
    assert result.equals(sample_data)

# ** Error Cases **

# Error Case 1: Unsupported data type
def test_unsupported_data_type():
    with pytest.raises(TypeError):
        sweep_nulls("not a dataframe", strategy='mean')
        sweep_nulls(pd.Series([1, 2, np.nan]), strategy='mean')

# Error Case 2: Invalid strategy
def test_invalid_strategy(sample_data):
    with pytest.raises(ValueError, match="Invalid strategy"):
        sweep_nulls(sample_data, strategy='invalid')

# Error Case 3: Non-numeric columns with numeric strategies
def test_non_numeric_with_numeric_strategy(sample_data):
    with pytest.raises(ValueError, match="Numeric strategies cannot be applied to non-numeric columns."):
        sweep_nulls(sample_data, strategy='mean', columns=['c'])

# Error Case 4: Invalid column name
def test_invalid_column_name(sample_data):
    with pytest.raises(KeyError, match="Invalid column name"):
        sweep_nulls(sample_data, strategy='mean', columns=['nonexistent_column'])

# Error Case 5: Missing `fill_value` for constant strategy
def test_missing_fill_value(sample_data):
    with pytest.raises(ValueError, match="`fill_value` must be provided for 'constant' strategy."):
        sweep_nulls(sample_data, strategy='constant')

# Error Case 6: Invalid `fill_value` type
def test_invalid_fill_value(sample_data):
    with pytest.raises(TypeError, match="Invalid `fill_value` type"):
        sweep_nulls(sample_data, strategy='constant', fill_value={"key": "value"})