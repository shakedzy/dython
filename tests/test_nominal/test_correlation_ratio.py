import numpy as np
from hypothesis import given, strategies as st, assume, settings, example

from dython.nominal import correlation_ratio


categories = st.text(alphabet=list("ABCDE"), min_size=1, max_size=1)


@st.composite
def categories_and_measurements(draw):
    n = draw(st.integers(min_value=2, max_value=30))
    category_lists = st.lists(categories, min_size=n, max_size=n)
    measurement_lists = st.lists(st.floats(), min_size=n, max_size=n)

    return draw(category_lists), draw(measurement_lists)


@given(c_m=categories_and_measurements())
def test_correlation_ratio_value_range(c_m):
    category, measurement = c_m

    corr_ratio = correlation_ratio(category, measurement)

    assert 0.0 <= corr_ratio <= 1.0 or np.isnan(corr_ratio)
