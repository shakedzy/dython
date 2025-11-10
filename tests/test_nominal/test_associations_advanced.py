import pytest
import numpy as np
import pandas as pd
from dython.nominal import associations, replot_last_associations


def test_associations_with_numerical_columns():
    """Test associations with numerical_columns parameter"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num1': [1, 2, 3, 4],
        'num2': [5, 6, 7, 8]
    })
    result = associations(df, numerical_columns=['num1', 'num2'], plot=False)
    assert 'corr' in result
    assert isinstance(result['corr'], pd.DataFrame)


def test_associations_with_numerical_columns_all():
    """Test associations with numerical_columns='all'"""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'num2': [5, 6, 7, 8]
    })
    result = associations(df, numerical_columns='all', plot=False)
    assert 'corr' in result


def test_associations_with_numerical_columns_auto():
    """Test associations with numerical_columns='auto'"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, numerical_columns='auto', plot=False)
    assert 'corr' in result


def test_associations_drop_samples():
    """Test associations with drop_samples nan strategy"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', None, 'a'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1'], nan_strategy='drop_samples', plot=False)
    assert 'corr' in result


def test_associations_drop_features():
    """Test associations with drop_features nan strategy"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', None, 'a'],
        'cat2': ['x', 'y', 'z', 'x'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], nan_strategy='drop_features', plot=False)
    assert 'corr' in result


def test_associations_drop_sample_pairs():
    """Test associations with drop_sample_pairs nan strategy"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', None, 'a'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1'], nan_strategy='drop_sample_pairs', plot=False)
    assert 'corr' in result


def test_associations_invalid_nan_strategy():
    """Test associations with invalid nan strategy"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'num1': [1, 2, 3, 4]
    })
    with pytest.raises(ValueError):
        associations(df, nominal_columns=['cat1'], nan_strategy='invalid', plot=False)


def test_associations_hide_rows():
    """Test associations with hide_rows parameter"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], hide_rows='cat1', plot=False)
    assert 'corr' in result
    assert 'cat1' not in result['corr'].index


def test_associations_hide_columns():
    """Test associations with hide_columns parameter"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], hide_columns='num1', plot=False)
    assert 'corr' in result
    assert 'num1' not in result['corr'].columns


def test_associations_hide_rows_list():
    """Test associations with hide_rows as list"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], hide_rows=['cat1', 'cat2'], plot=False)
    assert 'corr' in result


def test_associations_hide_columns_list():
    """Test associations with hide_columns as list"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], hide_columns=['cat1', 'num1'], plot=False)
    assert 'corr' in result


def test_associations_display_rows_single():
    """Test associations with display_rows as single column"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], display_rows=['cat1'], plot=False)
    assert 'corr' in result
    assert 'cat1' in result['corr'].index


def test_associations_display_columns_single():
    """Test associations with display_columns as single column"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], display_columns='num1', plot=False)
    assert 'corr' in result


def test_associations_with_datetime():
    """Test associations with datetime columns"""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=4),
        'num1': [1, 2, 3, 4],
        'cat1': ['a', 'b', 'c', 'a']
    })
    result = associations(df, nominal_columns=['cat1'], plot=False)
    assert 'corr' in result


def test_associations_with_categorical_dtype():
    """Test associations with pandas categorical dtype"""
    df = pd.DataFrame({
        'cat1': pd.Categorical(['a', 'b', 'c', 'a']),
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns='auto', plot=False)
    assert 'corr' in result


def test_associations_with_categorical_nan():
    """Test associations with categorical dtype and NaN values"""
    df = pd.DataFrame({
        'cat1': pd.Categorical(['a', 'b', None, 'a']),
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1'], nan_strategy='replace', nan_replace_value='missing', plot=False)
    assert 'corr' in result


def test_associations_clustering():
    """Test associations with clustering enabled"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'] * 3,
        'cat2': ['x', 'y', 'x', 'y'] * 3,
        'num1': list(range(12)),
        'num2': list(range(12, 24))
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], clustering=True, plot=False)
    assert 'corr' in result


def test_associations_mark_columns():
    """Test associations with mark_columns enabled"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1'], mark_columns=True, plot=False)
    assert 'corr' in result
    assert any('(nom)' in str(col) for col in result['corr'].columns)


def test_associations_theil():
    """Test associations with Theil's U"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y']
    })
    result = associations(df, nominal_columns='all', nom_nom_assoc='theil', plot=False)
    assert 'corr' in result


def test_associations_spearman():
    """Test associations with Spearman correlation"""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'num2': [5, 6, 7, 8]
    })
    result = associations(df, nominal_columns=None, num_num_assoc='spearman', plot=False)
    assert 'corr' in result


def test_associations_kendall():
    """Test associations with Kendall correlation"""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'num2': [5, 6, 7, 8]
    })
    result = associations(df, nominal_columns=None, num_num_assoc='kendall', plot=False)
    assert 'corr' in result


def test_associations_custom_nom_nom():
    """Test associations with custom nominal-nominal function"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y']
    })
    
    def custom_assoc(x, y):
        return 0.5
    
    result = associations(df, nominal_columns='all', nom_nom_assoc=custom_assoc, plot=False)
    assert 'corr' in result


def test_associations_custom_nom_nom_asymmetric():
    """Test associations with custom asymmetric nominal-nominal function"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y']
    })
    
    def custom_assoc(x, y):
        return 0.5
    
    result = associations(df, nominal_columns='all', nom_nom_assoc=custom_assoc, symmetric_nom_nom=False, plot=False)
    assert 'corr' in result


def test_associations_custom_num_num():
    """Test associations with custom numerical-numerical function"""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'num2': [5, 6, 7, 8]
    })
    
    def custom_corr(x, y):
        return 0.9
    
    result = associations(df, nominal_columns=None, num_num_assoc=custom_corr, plot=False)
    assert 'corr' in result


def test_associations_custom_num_num_asymmetric():
    """Test associations with custom asymmetric numerical-numerical function"""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'num2': [5, 6, 7, 8]
    })
    
    def custom_corr(x, y):
        return 0.9
    
    result = associations(df, nominal_columns=None, num_num_assoc=custom_corr, symmetric_num_num=False, plot=False)
    assert 'corr' in result


def test_associations_custom_nom_num():
    """Test associations with custom nominal-numerical function"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'num1': [1, 2, 3, 4]
    })
    
    def custom_assoc(x, y):
        return 0.7
    
    result = associations(df, nominal_columns=['cat1'], nom_num_assoc=custom_assoc, plot=False)
    assert 'corr' in result


def test_associations_cramers_v_no_bias_correction():
    """Test associations with Cramer's V without bias correction"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'cat2': ['x', 'y', 'x', 'y']
    })
    result = associations(df, nominal_columns='all', cramers_v_bias_correction=False, plot=False)
    assert 'corr' in result


def test_replot_last_associations():
    """Test replot_last_associations function"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'num1': [1, 2, 3, 4]
    })
    # First create an association plot
    associations(df, nominal_columns=['cat1'], plot=False)
    
    # Now replot
    ax = replot_last_associations(plot=False)
    assert ax is not None


def test_replot_last_associations_without_prior():
    """Test replot_last_associations without prior associations call"""
    from dython.nominal import _ASSOC_PLOT_PARAMS
    _ASSOC_PLOT_PARAMS.clear()
    
    with pytest.raises(RuntimeError):
        replot_last_associations(plot=False)


def test_associations_compute_only():
    """Test associations with compute_only flag"""
    df = pd.DataFrame({
        'cat1': ['a', 'b', 'c', 'a'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1'], compute_only=True)
    assert 'corr' in result
    assert result['ax'] is None


def test_associations_single_value_column():
    """Test associations with single-value column"""
    df = pd.DataFrame({
        'cat1': ['a', 'a', 'a', 'a'],
        'cat2': ['x', 'y', 'x', 'y'],
        'num1': [1, 2, 3, 4]
    })
    result = associations(df, nominal_columns=['cat1', 'cat2'], plot=False)
    assert 'corr' in result

