import functools
import pytest
import pandas as pd
import matplotlib
from sklearn import datasets

from dython.nominal import associations


@pytest.fixture(autouse=True)
def disable_plot(monkeypatch):
    # Patch plt.show to not halt testing flow, by making it not block
    # function execution.
    # patch = functools.partial(matplotlib.pyplot.show, block=False)
    def patch(): pass
    monkeypatch.setattr(matplotlib.pyplot, "show", patch)


def test_return_type_check(iris_df):
    assoc = associations(iris_df)

    assert isinstance(assoc, dict), 'associations should return a dict'
    assert 'corr' in assoc, 'associations should return a dict containing "corr" key'
    assert 'ax' in assoc, 'associations should return a dict containing "ax" key'

    assert isinstance(assoc['corr'], pd.DataFrame), 'assoc["corr"] should be a pandas DataFrame'
    assert isinstance(assoc['ax'], matplotlib.axes.Axes), 'assoc["ax"] should be a matplotlib Axes'


def test_dimension_check(iris_df):
    corr = associations(iris_df)['corr']
    corr_shape = corr.shape
    iris_shape = iris_df.shape

    assert corr_shape[0] == corr_shape[1], 'association matrix has wrong shape'
    assert corr_shape[1] == iris_shape[1], 'association matrix has different shape from input data'


def test_single_value_zero_association(iris_df):
    SV_COL = 1
    iris_df.iloc[:, SV_COL] = 42

    corr = associations(iris_df)['corr']

    assert (corr.iloc[:, SV_COL] == 0).all(), 'single-value variable should have zero association value'
    assert (corr.iloc[SV_COL, :] == 0).all(), 'single-value variable should have zero association value'


def test_bad_nom_nom_assoc_parameter(iris_df):
    with pytest.raises(ValueError, match='is not a supported'):
        associations(iris_df, nom_nom_assoc='bad_parameter_name')


def test_bad_num_num_assoc_parameter(iris_df):
    with pytest.raises(ValueError, match='is not a supported'):
        associations(iris_df, num_num_assoc='bad_parameter_name')


def test_compute_only_ax_is_none(iris_df):
    assoc = associations(iris_df, compute_only = True)

    assert assoc['ax'] is None, 'associations with compute_only should return a None value for "ax" key'


def test_mark_columns(iris_df):
    corr = associations(iris_df, mark_columns = True)['corr']

    assert '(con)' in corr.index[0], "first column should contain (con) mark if iris_df is used"


