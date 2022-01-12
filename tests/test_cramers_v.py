import pytest

from dython.nominal import cramers_v


def test_cramers_v_check(iris_df):
    x = iris_df['extra']
    y = iris_df['target']

    # Note: this measure is symmetric
    assert cramers_v(x, y) == pytest.approx(0.14201914309546954)
    assert cramers_v(y, x) == pytest.approx(0.14201914309546954)

    assert cramers_v(x, x) == pytest.approx(1.0)
