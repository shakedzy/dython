import random
import numpy as np


def reorder_by_random_weighted_sampling(numbers, seed=None):
    """
    Reorder a list of numbers as if they were sampled using weighted-sampling
    with no replacement. Technically, using numpy.random.choice will be a better
    option, but I really like this algorithm, and thought it's worth implementing
    it. It is also quite easy to implement in other languages, where numpy is not
    an option.

    :param numbers: a list of numbers
    :param seed: an optional integer to be used as seed for randomness
    :return: the numbers supplied reordered as if they were sampled by weight
    """
    if seed:
        random.seed(seed)
    r_numbers = [np.exp(random.random(), 1.0/x) for x in numbers]
    reordered = [x for _,x in sorted(zip(r_numbers, numbers), reverse=True)]
    return reordered


def weighted_sampling(numbers, k=1, with_replacement=False):
    """
    Return k numbers from a weighted-sampling over the supplied numbers

    :param numbers: List or np.ndarray of numbers
    :param k: Integer. How many numbers to sample. Choosing k=None will yield a single number
    :param with_replacement: Boolean. Allow replacement or not
    :return: k numbers sampled from numbers
    """
    sampled = np.random.choice(numbers, size=k, replace=with_replacement)
    if isinstance(numbers, list) and k is not None:
        sampled = sampled.tolist()
    return sampled


def boltzmann_sampling(numbers, k=1, with_replacement=False):
    """
    Return k numbers from a boltzmann-sampling over the supplied numbers

    :param numbers: List or np.ndarray of numbers
    :param k: Integer. How many numbers to sample. Choosing k=None will yield a single number
    :param with_replacement: Boolean. Allow replacement or not
    :return: k numbers sampled from numbers
    """
    exp_func = np.vectorize(lambda x: np.exp(x))
    exp_numbers = exp_func(numbers)
    exp_sum = exp_numbers.sum()
    scaling_func = np.vectorize(lambda x: x/exp_sum)
    b_numbers = scaling_func(exp_numbers)
    return weighted_sampling(b_numbers, k=k, with_replacement=with_replacement)
