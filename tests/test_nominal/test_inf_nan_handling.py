import pytest
import numpy as np
import pandas as pd
from dython.nominal import associations, theils_u, correlation_ratio, cramers_v


def test_theils_u_single_category():
    """Test Theil's U with single category returns 1"""
    # When x and y are completely determined by each other
    x = ['a'] * 100
    y = ['a'] * 100
    
    result = theils_u(x, y)
    assert result == 1.0


def test_theils_u_with_replace_strategy():
    """Test Theil's U with replace nan strategy"""
    x = pd.Series(['a', 'b', 'c', None, 'a'])
    y = pd.Series(['x', 'y', 'z', 'w', 'x'])
    
    result = theils_u(x, y, nan_strategy='replace', nan_replace_value='missing')
    assert isinstance(result, float)


def test_correlation_ratio_with_replace_strategy():
    """Test correlation ratio with replace nan strategy"""
    categories = pd.Series(['a', 'b', 'c', None, 'a'])
    measurements = pd.Series([1.0, 2.0, 3.0, 4.0, 1.5])
    
    result = correlation_ratio(categories, measurements, nan_strategy='replace', nan_replace_value='missing')
    assert isinstance(result, float)


def test_correlation_ratio_zero_numerator():
    """Test correlation ratio with zero numerator"""
    # All measurements are the same
    categories = ['a', 'b', 'c', 'a']
    measurements = [5.0, 5.0, 5.0, 5.0]
    
    result = correlation_ratio(categories, measurements)
    assert result == 0.0


def test_correlation_ratio_precision_warning():
    """Test correlation ratio with values that need precision rounding"""
    # Create data that produces eta slightly > 1
    np.random.seed(42)
    categories = ['a'] * 50 + ['b'] * 50
    measurements = [1.0] * 50 + [2.0] * 50
    
    result = correlation_ratio(categories, measurements)
    assert 0.0 <= result <= 1.0


def test_cramers_v_without_bias_correction():
    """Test Cramer's V without bias correction"""
    x = ['a', 'b', 'c', 'a']
    y = ['x', 'y', 'x', 'y']
    
    result = cramers_v(x, y, bias_correction=False)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_associations_with_inf_nan_values():
    """Test associations with inf/nan producing columns"""
    df = pd.DataFrame({
        'cat1': ['a'] * 10,  # Single value
        'cat2': ['x', 'y'] * 5,
        'num1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    
    result = associations(df, nominal_columns=['cat1', 'cat2'], plot=False)
    assert 'corr' in result

