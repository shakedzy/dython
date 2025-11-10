import pytest
import numpy as np
import pandas as pd
from dython.nominal import numerical_encoding


def test_numerical_encoding_basic():
    """Test basic numerical encoding"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num': [1, 2, 3, 4]
    })
    result = numerical_encoding(df, nominal_columns=['cat1', 'cat2'])
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) >= len(df.columns)


def test_numerical_encoding_auto():
    """Test numerical encoding with auto detection"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'num': [1, 2, 3, 4]
    })
    result = numerical_encoding(df, nominal_columns='auto')
    assert isinstance(result, pd.DataFrame)


def test_numerical_encoding_all():
    """Test numerical encoding with all columns"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y']
    })
    result = numerical_encoding(df, nominal_columns='all')
    assert isinstance(result, pd.DataFrame)


def test_numerical_encoding_none():
    """Test numerical encoding with None nominal columns"""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'num2': [5, 6, 7, 8]
    })
    result = numerical_encoding(df, nominal_columns=None)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_numerical_encoding_single_value_with_numeric():
    """Test numerical encoding with single value column alongside numeric"""
    df = pd.DataFrame({
        'num': [1, 2, 3, 4],
        'cat1': ['a', 'a', 'a', 'a'],
        'cat2': ['x', 'y', 'x', 'y']
    })
    result = numerical_encoding(df, nominal_columns=['cat1', 'cat2'])
    assert isinstance(result, pd.DataFrame)
    # cat1 should be encoded as 0 (single value)
    assert 'cat1' in result.columns
    assert (result['cat1'] == 0).all()


def test_numerical_encoding_drop_single_label():
    """Test numerical encoding with drop_single_label=True"""
    df = pd.DataFrame({
        'cat1': ['a', 'a', 'a', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num': [1, 2, 3, 4]
    })
    result = numerical_encoding(df, nominal_columns=['cat1', 'cat2'], drop_single_label=True)
    assert isinstance(result, pd.DataFrame)
    assert 'cat1' not in result.columns


def test_numerical_encoding_return_dict():
    """Test numerical encoding with drop_fact_dict=False"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'a', 'b'],
        'num': [1, 2, 3, 4]
    })
    result = numerical_encoding(df, nominal_columns=['cat1'], drop_fact_dict=False)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], dict)


def test_numerical_encoding_with_nan_replace():
    """Test numerical encoding with nan replace strategy"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', None, 'a'],
        'num': [1, 2, 3, np.nan]
    })
    result = numerical_encoding(df, nominal_columns=['cat1'], nan_strategy='replace', nan_replace_value=0)
    assert isinstance(result, pd.DataFrame)
    assert not result.isnull().any().any()


def test_numerical_encoding_with_nan_drop_samples():
    """Test numerical encoding with nan drop_samples strategy"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', None, 'a'],
        'num': [1, 2, 3, 4]
    })
    result = numerical_encoding(df, nominal_columns=['cat1'], nan_strategy='drop_samples')
    assert isinstance(result, pd.DataFrame)
    assert len(result) < len(df)


def test_numerical_encoding_with_nan_drop_features():
    """Test numerical encoding with nan drop_features strategy"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', None, 'a'],
        'cat2': ['x', 'y', 'z', 'x'],
        'num': [1, 2, 3, 4]
    })
    result = numerical_encoding(df, nominal_columns=['cat1', 'cat2'], nan_strategy='drop_features')
    assert isinstance(result, pd.DataFrame)
    assert 'cat1' not in result.columns


def test_numerical_encoding_three_plus_values():
    """Test numerical encoding with more than two values (get_dummies)"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'd'],
        'num': [1, 2, 3, 4]
    })
    result = numerical_encoding(df, nominal_columns=['cat1'])
    assert isinstance(result, pd.DataFrame)
    # Should have dummy columns for cat1
    assert len(result.columns) > 2


def test_numerical_encoding_two_values():
    """Test numerical encoding with exactly two values (factorize)"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'a', 'b'],
        'num': [1, 2, 3, 4]
    })
    result, fact_dict = numerical_encoding(df, nominal_columns=['cat1'], drop_fact_dict=False)
    assert isinstance(result, pd.DataFrame)
    assert 'cat1' in fact_dict
    assert len(fact_dict['cat1']) == 2

