import numpy as np

__all__ = [
    'boltzmann_sampling',
    'weighted_sampling'
]


def _w_sampling(numbers, k, with_replacement, force_to_list):
    sampled = np.random.choice(numbers, size=k, replace=with_replacement)
    if (isinstance(numbers, list) or force_to_list) and k is not None:
        sampled = sampled.tolist()
    return sampled


def weighted_sampling(numbers, k=1, with_replacement=False):
    """
    Return k numbers from a weighted-sampling over the supplied numbers

    Parameters:
    -----------
    numbers : List or np.ndarray
        numbers to sample
    k : int, default = 1
        How many numbers to sample. Choosing `k=None` will yield a single
        number
    with_replacement : Boolean, default = False
        Allow replacement or not

    Returns:
    --------
    List, np.ndarray or a single number (depending on the input)
    """
    return _w_sampling(numbers, k, with_replacement, force_to_list=False)


def boltzmann_sampling(numbers, k=1, with_replacement=False):
    """
    Return k numbers from a boltzmann-sampling over the supplied numbers

    Parameters:
    -----------
    numbers : List or np.ndarray
        numbers to sample
    k : int, default = 1
        How many numbers to sample. Choosing `k=None` will yield a single
        number
    with_replacement : Boolean, default = False
        Allow replacement or not

    Returns:
    --------
    List, np.ndarray or a single number (depending on the input)
    """
    exp_func = np.vectorize(lambda x: np.exp(x))
    exp_numbers = exp_func(numbers)
    exp_sum = exp_numbers.sum()
    scaling_func = np.vectorize(lambda x: x / exp_sum)
    b_numbers = scaling_func(exp_numbers)
    return _w_sampling(b_numbers,
                       k=k,
                       with_replacement=with_replacement,
                       force_to_list=isinstance(numbers, list))
