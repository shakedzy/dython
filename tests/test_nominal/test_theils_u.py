import pytest
from hypothesis import given, strategies as st

from dython.nominal import theils_u


def test_theils_u_check(iris_df):
    x = iris_df["extra"]
    y = iris_df["target"]

    # Note: this measure is not symmetric
    assert theils_u(x, y) == pytest.approx(0.02907500150218738)
    assert theils_u(y, x) == pytest.approx(0.0424761859049835)


categories = st.text(alphabet=list("ABCDE"), min_size=1, max_size=1)


@given(x=st.lists(categories, min_size=2, max_size=30))
def test_theils_u_identity(x):
    assert theils_u(x, x) == pytest.approx(1.0)


@st.composite
def two_categorical_lists(draw):
    n = draw(st.integers(min_value=2, max_value=30))
    categorical_lists = st.lists(categories, min_size=n, max_size=n)

    return draw(categorical_lists), draw(categorical_lists)


@given(x_y=two_categorical_lists())
def test_theils_u_value_range(x_y):
    x, y = x_y

    u_xy = theils_u(x, y)

    assert 0.0 <= u_xy <= 1.0
