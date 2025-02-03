import pytest
import pandas as pd
import numpy as np
from datamop.column_scaler import column_scaler

# Fixture for test data
@pytest.fixture
def one_column_df():
    """Return DataFrame with one column of numeric values. Used for testing."""
    return pd.DataFrame({"price": [25, 50, 75]})

@pytest.fixture
def one_column_df_float():
    """Return DataFrame with one column of floating values. Used for testing."""
    return pd.DataFrame({"price": [25.0, 50.0, 75.0]})

@pytest.fixture
def single_val_df():
    """Return DataFrame with one column with single repeated value. Used for testing."""
    return pd.DataFrame({"price": [10, 10, 10]})

@pytest.fixture
def empty_df():
    """Return empty DataFrame. Used for testing."""
    return pd.DataFrame()

@pytest.fixture
def non_numeric_df():
    """Return DataFrame with one column with non-numeric data. Used for testing."""
    return pd.DataFrame({"price": ["apple", "pear", "banana"]})

# Expected use case tests
def test_minmax_scaling_default(one_column_df):
    """Test min-max scaling with default new_min=0 and new_max=1. Use float values."""
    scaled_df = column_scaler(one_column_df, column="price", method="minmax")
    expected = [0.0, 0.5, 1.0]
    assert scaled_df["price"].tolist() == expected

def test_minmax_scaling_default_float(one_column_df_float):
    """Test min-max scaling with default new_min=0 and new_max=1."""
    scaled_df = column_scaler(one_column_df_float, column="price", method="minmax")
    expected = [0.0, 0.5, 1.0]
    assert scaled_df["price"].tolist() == expected

def test_minmax_scaling_custom(one_column_df):
    """Test min-max scaling with custom new_min=10 and new_max=20."""
    scaled_df = column_scaler(one_column_df, column="price", method="minmax", new_min=10, new_max=20)
    expected = [10.0, 15.0, 20.0]
    assert scaled_df["price"].tolist() == expected

def test_standard_scaling(one_column_df):
    """Test standard scaling, ensure mean is 0 and std is 1."""
    scaled_df = column_scaler(one_column_df, column="price", method="standard")
    assert np.isclose(scaled_df["price"].mean(), 0.0)
    assert np.isclose(scaled_df["price"].std(), 1.0)

def test_inplace_false(one_column_df):
    """Test when inplace is set to `False`."""
    scaled_df = column_scaler(one_column_df, column="price", method="minmax", inplace=False)
    expected = [0.0, 0.5, 1.0]
    assert one_column_df["price"].tolist() == [25, 50, 75]
    assert scaled_df["price_scaled"].tolist() == expected

# Edge case tests
def test_single_value_column_minmax(single_val_df):
    """Test minmax scaling with column with single repeated values."""
    with pytest.warns(UserWarning, match="Single-value column detected"):
        scaled_df = column_scaler(single_val_df, column="price", method="minmax", new_min=10, new_max=20)
    expected = [15.0, 15.0, 15.0]
    assert scaled_df["price"].tolist() == expected

def test_single_value_column_standard(single_val_df):
    """Test standard scaling with column with single repeated values to prevent division by zero."""
    with pytest.warns(UserWarning, 
                      match="Standard deviation is zero"):
        scaled_df = column_scaler(single_val_df, column="price", method="standard")
    
    expected = [0, 0, 0]
    assert scaled_df["price"].tolist() == expected

def test_empty_dataframe(empty_df):
    """Test scaling on empty DataFrame."""
    with pytest.warns(UserWarning, match="Empty DataFrame detected"):
        scaled_df = column_scaler(empty_df, column="price", method="minmax")
    assert scaled_df.empty

def test_column_with_nan():
    """Test scaling when there are NaN values in the column."""
    nan_df = pd.DataFrame({"price": [10, np.nan, 30]})
    with pytest.warns(UserWarning, match="NaN value detected in column"):
        scaled_df = column_scaler(nan_df, column="price", method="minmax")

    expected = [0.0, np.nan, 1.0]
    assert np.allclose(scaled_df["price"], expected, equal_nan=True)


# Erroneous case tests

def test_non_numeric_column(non_numeric_df):
    """Test scaling with a non-numeric column."""
    with pytest.raises(ValueError, match="Column must have numeric values."):
        column_scaler(non_numeric_df, column="price", method="minmax")

def test_invalid_column_name(one_column_df):
    """Test scaling when column name is invalid."""
    with pytest.raises(KeyError, match="Column not found in the DataFrame."):
        column_scaler(one_column_df, column="invalid column", method="minmax")

def test_invalid_method(one_column_df):
    """Test scaling with invalid method."""
    with pytest.raises(ValueError, match="Invalid method. Method should be `minmax` or `standard`."):
        column_scaler(one_column_df, column="price", method="invalid")


def test_invalid_data_type():
    """Test scaling with an invalid data type."""
    invalid_data = [1, 4, 8]
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame."):
        column_scaler(invalid_data, column="price", method="minmax")

def test_minmax_boundary(one_column_df):
    """Test scaling with `new_min` greater than `new_max`."""
    with pytest.raises(ValueError, match="`new_min` cannot be greater than `new_max`."):
        column_scaler(one_column_df, column="price", method="minmax", new_min=20, new_max=10)
