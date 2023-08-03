import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Any, Tuple


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
        plt.close()


def convert(
        data: ArrayLike[Any],
        to: str,
        copy: bool = True
) -> ArrayLike:

    converted = None
    if to == "array":
        if isinstance(data, np.ndarray):
            converted = data.copy() if copy else data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.values()
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
        return converted


def remove_incomplete_samples(
        x: ArrayLike[Any],
        y: ArrayLike[Any]
)-> ArrayLike:

    x = [v if v is not None else np.nan for v in x]
    y = [v if v is not None else np.nan for v in y]
    arr = np.array([x, y]).transpose()
    arr = arr[~np.isnan(arr).any(axis=1)].transpose()
    if isinstance(x, list):
        return arr[0].tolist(), arr[1].tolist()
    else:
        return arr[0], arr[1]


def replace_nan_with_value(
        x: ArrayLike[Any],
        y: ArrayLike[Any],
        value: Any
) -> Tuple[NDArray, NDArray]:
    x = np.array(
        [v if v == v and v is not None else value for v in x]
    )  # NaN != NaN
    y = np.array([v if v == v and v is not None else value for v in y])
    return x, y
