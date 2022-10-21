import pytest
import numpy as np
from dython.sampling import boltzmann_sampling, weighted_sampling


@pytest.fixture(params=["list", "array"])
def population(request):
    if request.param == "list":
        return [0.0, 1.0, 2.0, 3.0, 4.0]
    elif request.param == "array":
        return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


parametrize_sampling_funcs = pytest.mark.parametrize(
    "func", [boltzmann_sampling, weighted_sampling]
)


@parametrize_sampling_funcs
def test_k_none(func, population):
    result = func(population, k=None)
    assert type(result) is np.float64


@parametrize_sampling_funcs
@pytest.mark.parametrize("k", [1, 2])
def test_k_number(func, population, k):
    result = func(population, k=k)
    assert type(result) == type(
        population
    ), "Sampling with k != None should return same type as input"
