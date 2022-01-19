import pytest
import functools
import numpy as np
from hypothesis import given, strategies as st, assume, settings, example

from dython.nominal import cramers_v


# "Patch" pytest.approx to increase its tolerance range
approx = functools.partial(pytest.approx, abs = 1e-6, rel=1e-6)


def test_cramers_v_check(iris_df):
    x = iris_df['extra']
    y = iris_df['target']

    # Note: this measure is symmetric
    assert cramers_v(x, y) == pytest.approx(0.14201914309546954)
    assert cramers_v(y, x) == pytest.approx(0.14201914309546954)


categories = st.text(alphabet=list("ABCDE"), min_size=1, max_size=1)


@st.composite
def two_categorical_lists(draw):
    n = draw(st.integers(min_value=2, max_value=30))
    categorical_lists = st.lists(categories, min_size = n, max_size = n)

    return draw(categorical_lists), draw(categorical_lists)


@given(x_y = two_categorical_lists())
def test_cramers_v_value_range(x_y):
    x, y = x_y

    v_xy = cramers_v(x, y)

    assume(not np.isnan(v_xy))

    # 0.0 <= v_xy <= 1.0 is false when v_xy == 1.00000000000004
    # hence this weird-looking assertion, to avoid hypothesis saying it's "flaky"
    assert v_xy == pytest.approx(0.0) or 0.0 < v_xy < 1.0 or v_xy == pytest.approx(1.0)


@given(x_y = two_categorical_lists())
@settings(deadline=1000)
def test_cramers_v_symmetry(x_y):
    x, y = x_y
    v_xy = cramers_v(x, y)
    v_yx = cramers_v(y, x)

    # Can be overridden by passing nan_ok = True to
    # pytest.approx, but this feels more appropriate
    assume(not np.isnan(v_xy) and not np.isnan(v_yx))

    assert approx(v_xy) == approx(v_yx)
