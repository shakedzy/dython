import pytest
import numpy as np
import pandas as pd
from dython.nominal import conditional_entropy


def test_conditional_entropy_basic():
    """Test basic conditional entropy calculation"""
    x = [1, 1, 2, 2, 3, 3]
    y = ['a', 'a', 'b', 'b', 'c', 'c']
    result = conditional_entropy(x, y)
    assert isinstance(result, float)
    assert result >= 0


def test_conditional_entropy_with_drop_strategy():
    """Test conditional entropy with drop nan strategy"""
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, np.nan])
    y = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    result = conditional_entropy(x, y, nan_strategy='drop')
    assert isinstance(result, float)
    assert result >= 0


def test_conditional_entropy_with_replace_strategy():
    """Test conditional entropy with replace nan strategy"""
    x = [1, 1, 2, 2, 3, None]
    y = ['a', 'a', 'b', 'b', 'c', 'c']
    result = conditional_entropy(x, y, nan_strategy='replace', nan_replace_value=0)
    assert isinstance(result, float)
    assert result >= 0


def test_conditional_entropy_custom_log_base():
    """Test conditional entropy with custom log base"""
    x = [1, 1, 2, 2, 3, 3]
    y = ['a', 'a', 'b', 'b', 'c', 'c']
    result = conditional_entropy(x, y, log_base=2)
    assert isinstance(result, float)
    assert result >= 0


def test_conditional_entropy_with_pandas():
    """Test conditional entropy with pandas Series"""
    x = pd.Series([1, 1, 2, 2, 3, 3])
    y = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    result = conditional_entropy(x, y)
    assert isinstance(result, float)
    assert result >= 0

