import numpy as np
import pandas as pd
from typing import Sequence, Any, TypedDict, Protocol
from matplotlib.axes._axes import Axes


Number = int | float
OneDimArray = Sequence[Number | str] | pd.Series | np.ndarray[Any, np.dtype[np.int64] | np.dtype[np.float64] | np.dtype[np.str_]]
TwoDimArray = np.ndarray[Any, np.dtype[np.int64] | np.dtype[np.float64] | np.dtype[np.str_]] | pd.DataFrame 


class AssociationsResult(TypedDict):
    corr: pd.DataFrame
    ax: Axes | None


class SingleMethodResult(TypedDict):
    x: float
    y: float
    val: float


class SingleCurveResult(Protocol):
    auc: dict[str, float]
    def __getitem__(self, key: str) -> SingleMethodResult: ...


class MetricGraphResult(TypedDict):
    metrics: dict[str, SingleCurveResult] | SingleCurveResult
    ax: Axes