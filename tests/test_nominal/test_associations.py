import pytest
import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn import datasets
from datetime import datetime, timedelta

from dython.nominal import associations, correlation_ratio


def test_return_type_check(iris_df):
    assoc = associations(iris_df)

    assert isinstance(assoc, dict), "associations should return a dict"
    assert (
        "corr" in assoc
    ), 'associations should return a dict containing "corr" key'
    assert (
        "ax" in assoc
    ), 'associations should return a dict containing "ax" key'

    assert isinstance(
        assoc["corr"], pd.DataFrame
    ), 'assoc["corr"] should be a pandas DataFrame'
    assert isinstance(
        assoc["ax"], matplotlib.axes.Axes
    ), 'assoc["ax"] should be a matplotlib Axes'


def test_dimension_check(iris_df):
    corr = associations(iris_df)["corr"]
    corr_shape = corr.shape
    iris_shape = iris_df.shape

    assert corr_shape[0] == corr_shape[1], "association matrix has wrong shape"
    assert (
        corr_shape[1] == iris_shape[1]
    ), "association matrix has different shape from input data"


def test_single_value_zero_association(iris_df):
    SV_COL = 1
    iris_df.iloc[:, SV_COL] = 42

    corr = associations(iris_df)["corr"]

    assert (
        corr.iloc[:, SV_COL] == 0
    ).all(), "single-value variable should have zero association value"
    assert (
        corr.iloc[SV_COL, :] == 0
    ).all(), "single-value variable should have zero association value"


def test_bad_nom_nom_assoc_parameter(iris_df):
    with pytest.raises(ValueError, match="is not a supported"):
        associations(iris_df, nom_nom_assoc="bad_parameter_name")


def test_bad_num_num_assoc_parameter(iris_df):
    with pytest.raises(ValueError, match="is not a supported"):
        associations(iris_df, num_num_assoc="bad_parameter_name")


def test_compute_only_ax_is_none(iris_df):
    assoc = associations(iris_df, compute_only=True)

    assert (
        assoc["ax"] is None
    ), 'associations with compute_only should return a None value for "ax" key'


def test_mark_columns(iris_df):
    corr = associations(iris_df, mark_columns=True)["corr"]

    assert (
        "(con)" in corr.index[0]
    ), "first column should contain (con) mark if iris_df is used"


def test_udf(iris_df):
    def pr(x, y):
        return ss.pearsonr(x, y)[0]

    corr1 = associations(
        iris_df,
        plot=False,
        num_num_assoc="pearson",
        nom_num_assoc="correlation_ratio",
    )["corr"]
    corr2 = associations(
        iris_df, plot=False, num_num_assoc=pr, nom_num_assoc=correlation_ratio
    )["corr"]
    assert corr1.compare(
        corr2
    ).empty, (
        "Computation of built-in measures of associations differs from UDFs"
    )


def test_datetime_data():
    dt = datetime(2020, 12, 1)
    end = datetime(2020, 12, 2)
    step = timedelta(seconds=5)
    result = []
    while dt < end:
        result.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        dt += step

    nums = list(range(len(result)))
    df = pd.DataFrame(
        {"dates": result, "up": nums, "down": sorted(nums, reverse=True)}
    )
    df["dates"] = pd.to_datetime(
        df["dates"], format="%Y-%m-%d %H:%M:%S"
    )  # without this, this column is considered as object rather than dates

    correct_corr = pd.DataFrame(
        columns=["dates", "up", "down"],
        index=["dates", "up", "down"],
        data=[[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
    )
    corr = associations(df, plot=False)["corr"]
    assert corr.compare(
        correct_corr
    ).empty, f"datetime associations are incorrect. Test should have returned an empty dataframe, received: {corr.head()}"


def test_category_nan_replace(iris_df):
    iris_df["extra"] = iris_df["extra"].astype("category")
    iris_df.loc[5, "extra"] = np.nan
    try:
        associations(iris_df, nan_strategy="replace")
    except TypeError as exception:
        assert (
            False
        ), f"nan_strategy='replace' with a pandas.CategoricalDtype column raised an exception {exception}"
