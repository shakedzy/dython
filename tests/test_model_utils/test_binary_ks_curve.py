import pytest
import numpy as np
from dython.model_utils import _binary_ks_curve


def test_binary_ks_curve_basic():
    """Test basic binary_ks_curve functionality"""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_probas = np.array([0.1, 0.9, 0.3, 0.8, 0.7, 0.2])
    
    thresholds, pct1, pct2, ks_stat, max_dist, classes = _binary_ks_curve(y_true, y_probas)
    
    assert len(thresholds) > 0
    assert len(pct1) == len(thresholds)
    assert len(pct2) == len(thresholds)
    assert 0 <= ks_stat <= 1
    assert len(classes) == 2


def test_binary_ks_curve_multiclass_error():
    """Test binary_ks_curve with more than 2 classes"""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_probas = np.array([0.1, 0.5, 0.9, 0.2, 0.6, 0.8])
    
    with pytest.raises(ValueError):
        _binary_ks_curve(y_true, y_probas)


def test_binary_ks_curve_thresholds_start_with_zero():
    """Test that thresholds start with 0"""
    y_true = np.array([0, 1, 0, 1])
    y_probas = np.array([0.2, 0.8, 0.3, 0.9])
    
    thresholds, _, _, _, _, _ = _binary_ks_curve(y_true, y_probas)
    
    assert thresholds[0] == 0.0


def test_binary_ks_curve_thresholds_end_with_one():
    """Test that thresholds end with 1"""
    y_true = np.array([0, 1, 0, 1])
    y_probas = np.array([0.2, 0.8, 0.3, 0.7])
    
    thresholds, _, _, _, _, _ = _binary_ks_curve(y_true, y_probas)
    
    assert thresholds[-1] == 1.0


def test_binary_ks_curve_with_edge_probabilities():
    """Test binary_ks_curve with probabilities at 0 and 1"""
    y_true = np.array([0, 1, 0, 1])
    y_probas = np.array([0.0, 1.0, 0.1, 0.9])
    
    thresholds, pct1, pct2, ks_stat, max_dist, _ = _binary_ks_curve(y_true, y_probas)
    
    assert len(thresholds) > 0
    assert thresholds[0] == 0.0
    assert thresholds[-1] == 1.0


def test_binary_ks_curve_data1_exhausted_first():
    """Test binary_ks_curve when data1 is exhausted before data2"""
    y_true = np.array([0, 0, 1, 1, 1])
    y_probas = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
    
    thresholds, pct1, pct2, ks_stat, max_dist, _ = _binary_ks_curve(y_true, y_probas)
    
    assert len(thresholds) > 0
    assert pct1[-1] == 1.0
    assert pct2[-1] == 1.0


def test_binary_ks_curve_data2_exhausted_first():
    """Test binary_ks_curve when data2 is exhausted before data1"""
    y_true = np.array([0, 0, 0, 1, 1])
    y_probas = np.array([0.6, 0.7, 0.8, 0.1, 0.2])
    
    thresholds, pct1, pct2, ks_stat, max_dist, _ = _binary_ks_curve(y_true, y_probas)
    
    assert len(thresholds) > 0
    assert pct1[-1] == 1.0
    assert pct2[-1] == 1.0


def test_binary_ks_curve_equal_values():
    """Test binary_ks_curve with equal probability values"""
    y_true = np.array([0, 0, 1, 1])
    y_probas = np.array([0.5, 0.5, 0.5, 0.5])
    
    thresholds, pct1, pct2, ks_stat, max_dist, _ = _binary_ks_curve(y_true, y_probas)
    
    assert len(thresholds) > 0


def test_binary_ks_curve_interleaved_values():
    """Test binary_ks_curve with interleaved probability values"""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_probas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    thresholds, pct1, pct2, ks_stat, max_dist, _ = _binary_ks_curve(y_true, y_probas)
    
    assert len(thresholds) > 0
    assert 0 <= ks_stat <= 1

