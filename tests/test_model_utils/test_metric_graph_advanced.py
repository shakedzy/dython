import pytest
import numpy as np
import matplotlib.pyplot as plt
from dython.model_utils import metric_graph


def test_metric_graph_invalid_metric():
    """Test metric_graph with invalid metric"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    with pytest.raises(ValueError):
        metric_graph(y_true, y_pred, metric='invalid', plot=False)


def test_metric_graph_none_metric():
    """Test metric_graph with None metric"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    with pytest.raises(ValueError):
        metric_graph(y_true, y_pred, metric=None, plot=False)


def test_metric_graph_multiclass_with_class_names():
    """Test metric_graph with multiclass and class names"""
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    
    result = metric_graph(y_true, y_pred, metric='roc', class_names=['A', 'B', 'C'], plot=False)
    assert 'A' in result
    assert 'B' in result
    assert 'C' in result


def test_metric_graph_multiclass_wrong_class_names_type():
    """Test metric_graph with multiclass and wrong class names type"""
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    
    # class_names as non-list/non-string for multiclass should raise error
    with pytest.raises((ValueError, TypeError)):
        metric_graph(y_true, y_pred, metric='roc', class_names=123, plot=False)


def test_metric_graph_multiclass_wrong_class_names_count():
    """Test metric_graph with multiclass and wrong number of class names"""
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    
    with pytest.raises(ValueError):
        metric_graph(y_true, y_pred, metric='roc', class_names=['A', 'B'], plot=False)


def test_metric_graph_pr_binary():
    """Test PR curve for binary classification"""
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.3, 0.8, 0.7, 0.2]
    
    result = metric_graph(y_true, y_pred, metric='pr', plot=False)
    assert 'auc' in result['0']
    assert 'naive' in result['0']['auc']


def test_metric_graph_pr_multiclass():
    """Test PR curve for multiclass"""
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    
    result = metric_graph(y_true, y_pred, metric='pr', plot=False)
    assert '0' in result


def test_metric_graph_with_colors_string():
    """Test metric_graph with colors as string"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    result = metric_graph(y_true, y_pred, metric='roc', colors='red', plot=False)
    assert 'ax' in result


def test_metric_graph_multiclass_no_micro():
    """Test metric_graph multiclass without micro"""
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    
    result = metric_graph(y_true, y_pred, metric='roc', micro=False, plot=False)
    assert '0' in result


def test_metric_graph_multiclass_no_macro():
    """Test metric_graph multiclass without macro"""
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    
    result = metric_graph(y_true, y_pred, metric='roc', macro=False, plot=False)
    assert '0' in result


def test_metric_graph_pr_multiclass_no_macro():
    """Test PR curve multiclass without macro (macro not applicable for PR)"""
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])
    
    result = metric_graph(y_true, y_pred, metric='pr', macro=False, plot=False)
    assert '0' in result


def test_metric_graph_binary_no_eopt():
    """Test metric_graph binary without eopt"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    result = metric_graph(y_true, y_pred, metric='roc', eopt=False, plot=False)
    assert result['0']['eopt']['val'] is None


def test_metric_graph_multiclass_force():
    """Test metric_graph with force_multiclass flag"""
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    
    result = metric_graph(y_true, y_pred, metric='roc', force_multiclass=True, plot=False)
    assert '0' in result
    assert '1' in result


def test_metric_graph_binary_2d_array():
    """Test metric_graph binary with 2D array"""
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    
    result = metric_graph(y_true, y_pred, metric='roc', plot=False)
    assert 'auc' in result['0']


def test_metric_graph_mismatched_shapes():
    """Test metric_graph with mismatched shapes"""
    y_true = [0, 1, 0]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    with pytest.raises(ValueError):
        metric_graph(y_true, y_pred, metric='roc', plot=False)


def test_metric_graph_with_ax():
    """Test metric_graph with provided ax"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    fig, ax = plt.subplots()
    result = metric_graph(y_true, y_pred, metric='roc', ax=ax, plot=False)
    assert result['ax'] == ax
    plt.close(fig)


def test_metric_graph_binary_1d_arrays():
    """Test metric_graph binary with 1D arrays"""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.2, 0.7, 0.3])
    
    result = metric_graph(y_true, y_pred, metric='roc', plot=False)
    assert '0' in result


def test_metric_graph_with_custom_params():
    """Test metric_graph with custom visualization parameters"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    result = metric_graph(
        y_true, y_pred, 
        metric='roc', 
        xlim=(0, 0.5),
        ylim=(0.5, 1.0),
        lw=3,
        ls='--',
        ms=15,
        fmt='.3f',
        legend='upper right',
        title='Custom Title',
        plot=False
    )
    assert 'ax' in result


def test_metric_graph_with_class_name_string():
    """Test metric_graph with class_names as string for binary"""
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.3, 0.8]
    
    result = metric_graph(y_true, y_pred, metric='roc', class_names='PositiveClass', plot=False)
    assert 'PositiveClass' in result

