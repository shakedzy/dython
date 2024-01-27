from pandas import DataFrame, Series
from typing import List, Union, Any
from numpy.typing import NDArray


Number = Union[int, float]
OneDimArray = Union[List[Number], NDArray, Series]
TwoDimArray = Union[NDArray, DataFrame]
