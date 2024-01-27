from pandas import DataFrame, Series
from typing import List, Union, Any
from numpy.typing import NDArray


Number = Union[int, float]
OneDimArray = Union[List[Number], NDArray, Series[Number]]
TwoDimArray = Union[NDArray, DataFrame]
