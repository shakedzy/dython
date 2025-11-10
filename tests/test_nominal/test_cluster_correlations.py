import pytest
import numpy as np
import pandas as pd
from dython.nominal import cluster_correlations, associations


def test_cluster_correlations_with_dataframe():
    """Test cluster correlations with DataFrame input"""
    corr_mat = pd.DataFrame(
        [[1.0, 0.8, 0.3],
         [0.8, 1.0, 0.4],
         [0.3, 0.4, 1.0]],
        columns=['A', 'B', 'C'],
        index=['A', 'B', 'C']
    )
    result, indices = cluster_correlations(corr_mat)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(indices, np.ndarray)
    assert result.shape == corr_mat.shape


def test_cluster_correlations_with_numpy():
    """Test cluster correlations with numpy array input"""
    corr_mat = np.array(
        [[1.0, 0.8, 0.3],
         [0.8, 1.0, 0.4],
         [0.3, 0.4, 1.0]]
    )
    result, indices = cluster_correlations(corr_mat)
    assert isinstance(result, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert result.shape == corr_mat.shape


def test_cluster_correlations_with_provided_indices():
    """Test cluster correlations with provided indices"""
    corr_mat = pd.DataFrame(
        [[1.0, 0.8, 0.3],
         [0.8, 1.0, 0.4],
         [0.3, 0.4, 1.0]],
        columns=['A', 'B', 'C'],
        index=['A', 'B', 'C']
    )
    indices = np.array([1, 1, 2])
    result, result_indices = cluster_correlations(corr_mat, indices)
    assert isinstance(result, pd.DataFrame)
    assert np.array_equal(result_indices, indices)


def test_cluster_correlations_larger_matrix():
    """Test cluster correlations with larger matrix"""
    n = 10
    corr_mat = np.random.rand(n, n)
    corr_mat = (corr_mat + corr_mat.T) / 2  # Make symmetric
    np.fill_diagonal(corr_mat, 1.0)
    
    result, indices = cluster_correlations(corr_mat)
    assert isinstance(result, np.ndarray)
    assert result.shape == corr_mat.shape

