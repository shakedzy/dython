import pytest
import matplotlib

from dython.data_utils import split_hist


@pytest.fixture(autouse=True)
def disable_plot(monkeypatch):
    # Patch plt.show to not halt testing flow, by making it not block
    # function execution.
    # patch = functools.partial(matplotlib.pyplot.show, block=False)
    def patch(): pass
    monkeypatch.setattr(matplotlib.pyplot, "show", patch)


def test_split_hist_check(iris_df):
    result = split_hist(iris_df, 'sepal length (cm)', 'target')

    assert isinstance(result, matplotlib.axes.Axes)
