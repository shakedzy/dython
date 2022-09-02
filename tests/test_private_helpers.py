import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from dython._private import (
    convert,
    remove_incomplete_samples,
    replace_nan_with_value,
)

# Make pandas not emit SettingWithCopyWarning
# SettingWithCopyWarning looks relatively safe to ignore,
# compare with DeprecationWarning that eventually needs attention.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters
pd.set_option("mode.chained_assignment", None)


@pytest.fixture
def iris_df():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    return df


@pytest.fixture(params=["str", "tuple", "dict"])
def bad_input(request):
    if request.param == "str":
        return "EXAMPLE STRING"

    if request.param == "tuple":
        return "EXAMPLE", "TUPLE"

    if request.param == "dict":
        return {1: "EXAMPLE", 2: "DICT"}


@pytest.mark.parametrize("output_type", ["list", "array", "dataframe"])
def test_convert_good_output_bad_input(bad_input, output_type):
    with pytest.raises(TypeError, match="cannot handle data conversion"):
        convert(bad_input, output_type)


def test_convert_bad_output(iris_df):
    with pytest.raises(ValueError, match="Unknown"):
        convert(iris_df, "bad_parameter")


@pytest.fixture
def x_y(iris_df):
    x = iris_df[iris_df.columns[0]]
    y = iris_df[iris_df.columns[1]]
    return x, y


def test_remove_incomplete_cases_one_nan_each(x_y):
    x, y = x_y
    x[0] = None
    y[1] = None

    x_, y_ = remove_incomplete_samples(x, y)

    assert len(x_) == len(y_) == len(x) - 2


def test_remove_incomplete_cases_all_nan(x_y):
    x, y = x_y
    x = [None for _ in x]

    x_, y_ = remove_incomplete_samples(x, y)
    assert len(x_) == len(y_) == 0


def test_replace_nan_one_nan_each(x_y):
    x, y = x_y
    x[0] = None
    y[1] = None

    x_, y_ = replace_nan_with_value(x, y, 1_000)

    assert len(x_) == len(y_) == len(y)
    assert x_[0] == y_[1] == 1_000


def test_replace_nan_all_nan(x_y):
    x, y = x_y
    x = [None for _ in x]

    x_, y_ = replace_nan_with_value(x, y, 1_000)

    assert all([elem == 1_000 for elem in x_])
