from matplotlib.axes._axes import Axes

from dython.data_utils import split_hist


def test_split_hist_check(iris_df):
    result = split_hist(iris_df, "sepal length (cm)", "target")

    assert isinstance(result, Axes)
