import pytest
from dython.data_utils import one_hot_encode


def test_one_hot_encode_check():
    lst = [0, 0, 2, 5]
    row = len(lst)
    col = max(lst) + 1

    result = one_hot_encode(lst)
    assert result.shape == (row, col)


def test_negative_input():
    lst = [-1, -5, 0, 3]

    with pytest.raises(ValueError, match='negative value'):
        one_hot_encode(lst)


def test_more_than_one_dimension():
    lst = [[0, 1], [2, 3]]

    with pytest.raises(ValueError, match="must only have one dimension"):
        one_hot_encode(lst)


def test_single_value():
    lst = [0, 0, 0, 0]

    with pytest.raises(ValueError):
        one_hot_encode(lst)