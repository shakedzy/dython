import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Optional, Any, Tuple, Union, List, Literal
from .typing import Number, OneDimArray


IS_JUPYTER: bool = False


def set_is_jupyter(force_to: Optional[bool] = None) -> None:
    global IS_JUPYTER
    if force_to is not None:
        IS_JUPYTER = force_to
    else:
        IS_JUPYTER = "ipykernel_launcher.py" in sys.argv[0]


def plot_or_not(plot: bool) -> None:
    if plot:
        plt.show()
    elif not plot and IS_JUPYTER:
        fig = plt.gcf()
        if fig:  
            plt.close(fig)  


def convert(
    data: Union[List[Number], NDArray, pd.DataFrame],
    to: Literal["array", "list", "dataframe"],
    copy: bool = True,
) -> Union[List[Number], NDArray, pd.DataFrame]:
    converted = None
    if to == "array":
        if isinstance(data, np.ndarray):
            converted = data.copy() if copy else data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.values()  # type: ignore
    elif to == "list":
        if isinstance(data, list):
            converted = data.copy() if copy else data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == "dataframe":
        if isinstance(data, pd.DataFrame):
            converted = data.copy(deep=True) if copy else data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError(
            "cannot handle data conversion of type: {} to {}".format(
                type(data), to
            )
        )
    else:
        return converted  # type: ignore


def remove_incomplete_samples(
    x: Union[List[Any], OneDimArray], y: Union[List[Any], OneDimArray]
) -> Tuple[Union[List[Any], OneDimArray], Union[List[Any], OneDimArray]]:
    x = [v if v is not None else np.nan for v in x]
    y = [v if v is not None else np.nan for v in y]
    arr = np.array([x, y]).transpose()
    arr = arr[~np.isnan(arr).any(axis=1)].transpose()
    if isinstance(x, list):
        return arr[0].tolist(), arr[1].tolist()
    else:
        return arr[0], arr[1]


def replace_nan_with_value(
    x: Union[List[Any], OneDimArray],
    y: Union[List[Any], OneDimArray],
    value: Any,
) -> Tuple[NDArray, NDArray]:
    x = np.array(
        [v if v == v and v is not None else value for v in x]
    )  # NaN != NaN
    y = np.array([v if v == v and v is not None else value for v in y])
    return x, y
