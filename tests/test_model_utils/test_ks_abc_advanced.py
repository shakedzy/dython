import pytest
import numpy as np
import matplotlib.pyplot as plt
from dython.model_utils import ks_abc


def test_ks_abc_basic():
    """Test basic ks_abc functionality"""
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.3, 0.8, 0.7, 0.2]
    
    result = ks_abc(y_true, y_pred, plot=False)
    assert 'abc' in result
    assert 'ks_stat' in result
    assert 'eopt' in result
    assert 'ax' in result


def test_ks_abc_mismatched_shapes():
    """Test ks_abc with mismatched shapes"""
    y_true = [0, 1, 0]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    with pytest.raises(ValueError):
        ks_abc(y_true, y_pred, plot=False)


def test_ks_abc_2d_binary():
    """Test ks_abc with 2D binary array"""
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    
    result = ks_abc(y_true, y_pred, plot=False)
    assert 'abc' in result


def test_ks_abc_single_column():
    """Test ks_abc with single column"""
    y_true = np.array([[1], [0], [1], [0]])
    y_pred = np.array([[0.9], [0.2], [0.7], [0.3]])
    
    result = ks_abc(y_true, y_pred, plot=False)
    assert 'abc' in result


def test_ks_abc_multiclass_error():
    """Test ks_abc with multiclass (should raise error)"""
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    
    with pytest.raises(ValueError):
        ks_abc(y_true, y_pred, plot=False)


def test_ks_abc_with_ax():
    """Test ks_abc with provided ax"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    fig, ax = plt.subplots()
    result = ks_abc(y_true, y_pred, ax=ax, plot=False)
    assert result['ax'] == ax
    plt.close(fig)


def test_ks_abc_with_custom_params():
    """Test ks_abc with custom visualization parameters"""
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.3, 0.8, 0.7, 0.2]
    
    result = ks_abc(
        y_true, y_pred,
        colors=('red', 'blue'),
        title='Custom KS Title',
        xlim=(0, 0.5),
        ylim=(0, 0.5),
        fmt='.3f',
        lw=3,
        legend='upper left',
        plot=False
    )
    assert 'abc' in result


def test_ks_abc_no_legend():
    """Test ks_abc without legend"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    result = ks_abc(y_true, y_pred, legend=None, plot=False)
    assert 'ax' in result


def test_ks_abc_with_filename(tmp_path):
    """Test ks_abc with filename"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    filename = tmp_path / "ks_plot.png"
    result = ks_abc(y_true, y_pred, filename=str(filename), plot=False)
    assert filename.exists()
    plt.close('all')


def test_ks_abc_abc_value_range():
    """Test that ABC value is in valid range"""
    y_true = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.15, 0.85]
    
    result = ks_abc(y_true, y_pred, plot=False)
    assert 0 <= result['abc'] <= 1


def test_ks_abc_ks_stat_value_range():
    """Test that KS statistic is in valid range"""
    y_true = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.15, 0.85]
    
    result = ks_abc(y_true, y_pred, plot=False)
    assert 0 <= result['ks_stat'] <= 1

