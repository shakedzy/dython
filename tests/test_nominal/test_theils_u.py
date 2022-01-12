import pytest

from dython.nominal import theils_u


def test_theils_u_check(iris_df):
    x = iris_df['extra']
    y = iris_df['target']

    # Note: this measure is not symmetric
    assert theils_u(x, y) == pytest.approx(0.02907500150218738)
    assert theils_u(y, x) == pytest.approx(0.0424761859049835)

    assert theils_u(x, x) == pytest.approx(1.0)
