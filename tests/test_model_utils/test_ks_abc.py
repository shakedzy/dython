import pytest
import numpy as np
import matplotlib

from dython.model_utils import ks_abc


@pytest.fixture(autouse=True)
def disable_plot(monkeypatch):
    # Patch plt.show to not halt testing flow, by making it not block
    # function execution.
    # patch = functools.partial(matplotlib.pyplot.show, block=False)
    def patch(): pass
    monkeypatch.setattr(matplotlib.pyplot, "show", patch)


@pytest.fixture
def y_true():
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


@pytest.fixture
def y_pred():
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def test_ks_abc_check_types(y_true, y_pred):
    result = ks_abc(y_true, y_pred)

    assert isinstance(result, dict), 'ks_abc should return dict'

    assert 'abc' in result, 'ks_abc should return dict containing "abc" key'
    assert isinstance(result['abc'], float), 'area between curves should be a float'

    assert 'ks_stat' in result, 'ks_abc should return dict containing "ks_stat" key'
    assert isinstance(result['ks_stat'], float), 'ks statistic should be a float'

    assert 'eopt' in result, 'ks_abc should return dict containing "eopt" key'
    assert isinstance(result['eopt'], float), 'estimated optimal threshold should be a float'

    assert 'ax' in result, 'ks_abc should return dict containing "ax" key'
    assert isinstance(result['ax'], matplotlib.axes.Axes)


def test_ks_abc_check_known_value(y_true, y_pred):
    result = ks_abc(y_true, y_pred)

    assert result['abc'] == pytest.approx(0.55)
    assert result['ks_stat'] == pytest.approx(1.0)
    assert result['eopt'] == pytest.approx(0.4)
