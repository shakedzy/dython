import numpy as np


def weighted_sampling(numbers, k=1, with_replacement=False, **kwargs):
    """
    Return k numbers from a weighted-sampling over the supplied numbers

    :param numbers: List or np.ndarray of numbers
    :param k: Integer. How many numbers to sample. Choosing k=None will yield a single number
    :param with_replacement: Boolean. Allow replacement or not
    :return: k numbers sampled from numbers. List or np.ndarray, depending on the input. if k=None, returns a single number
    """
    sampled = np.random.choice(numbers, size=k, replace=with_replacement)
    if (isinstance(numbers, list) or kwargs.get('to_list', False)) and k is not None:
        sampled = sampled.tolist()
    return sampled


def boltzmann_sampling(numbers, k=1, with_replacement=False):
    """
    Return k numbers from a boltzmann-sampling over the supplied numbers

    :param numbers: List or np.ndarray of numbers
    :param k: Integer. How many numbers to sample. Choosing k=None will yield a single number
    :param with_replacement: Boolean. Allow replacement or not
    :return: k numbers sampled from numbers. List or np.ndarray, depending on the input. if k=None, returns a single number
    """
    exp_func = np.vectorize(lambda x: np.exp(x))
    exp_numbers = exp_func(numbers)
    exp_sum = exp_numbers.sum()
    scaling_func = np.vectorize(lambda x: x/exp_sum)
    b_numbers = scaling_func(exp_numbers)
    return weighted_sampling(b_numbers, k=k, with_replacement=with_replacement, to_list=isinstance(numbers, list))
