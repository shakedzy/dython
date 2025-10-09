import pytest
import numpy as np
import pandas as pd

from dython.nominal import cluster_correlations


@pytest.fixture
def corr_example():
    return pd.DataFrame(
        np.array(
            [
                [1, 0.5, 0.7, 0.3],
                [0.5, 1, 0.8, 0.2],
                [0.7, 0.8, 1, 0.1],
                [0.3, 0.2, 0.1, 1],
            ]
        ),
        columns=list("ABCD"),
        index=list("ABCD"),
    )


def test_cluster_correlation_check_return_values(corr_example):
    result = cluster_correlations(corr_example)

    assert isinstance(result, tuple), "should return a tuple"

    sorted_corr, indices = result

    assert isinstance(
        sorted_corr, pd.DataFrame
    ), "sorted correlation should be a pd.DataFrame correlation matrix"
    assert isinstance(indices, np.ndarray), "indices should be a np.ndarray"
