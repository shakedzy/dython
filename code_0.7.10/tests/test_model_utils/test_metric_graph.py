import pytest
import numpy as np
from matplotlib.axes._axes import Axes
from dython.model_utils import metric_graph


@pytest.fixture
def y_true():
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


@pytest.fixture
def y_pred():
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def test_metric_graph_check_types(y_true, y_pred):
    result = metric_graph(y_true, y_pred, "roc")

    assert isinstance(result, dict), "metric_graph should return a dict"

    assert "ax" in result, 'metric_graph should return dict containing "ax" key'

    assert isinstance(result["ax"], Axes)


def test_metric_graph_bad_metric_parameter(y_true, y_pred):
    with pytest.raises(ValueError, match="Invalid metric"):
        metric_graph(y_true, y_pred, "bad_metric_param")
