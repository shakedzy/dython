import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, cast, overload, Type
from .typing import OneDimArray, TwoDimArray


IS_JUPYTER: bool = False


def set_is_jupyter(force_to: bool | None = None) -> None:
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


@overload
def convert(
    data: OneDimArray | TwoDimArray,
    to: Type[np.ndarray],
    copy: bool = True,
) -> np.ndarray:
    ...

@overload
def convert(
    data: OneDimArray | TwoDimArray,
    to: Type[list],
    copy: bool = True,
) -> list:
    ...

@overload
def convert(
    data: OneDimArray | TwoDimArray,
    to: Type[pd.DataFrame],
    copy: bool = True,
) -> pd.DataFrame:
    ...

def convert(
    data: OneDimArray | TwoDimArray,
    to: Type[list | pd.DataFrame | np.ndarray],
    copy: bool = True,
) -> list | pd.DataFrame | np.ndarray:
    
    converted = None

    if to == np.ndarray:
        if isinstance(data, np.ndarray):
            converted = data.copy() if copy else data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.values             
        converted = cast(np.ndarray, converted)

    elif to == list:
        if isinstance(data, list):
            converted = data.copy() if copy else data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
        
    elif to == pd.DataFrame:
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
        return converted                                # pyright: ignore[reportReturnType]


def remove_incomplete_samples(
    x: OneDimArray, 
    y: OneDimArray,
) -> tuple[OneDimArray, OneDimArray]:
    
    x = [v if v is not None else np.nan for v in x]     # pyright: ignore[reportAssignmentType]
    y = [v if v is not None else np.nan for v in y]     # pyright: ignore[reportAssignmentType]
    arr = np.array([x, y]).transpose()
    arr = arr[~np.isnan(arr).any(axis=1)].transpose()
    if isinstance(x, list):
        return arr[0].tolist(), arr[1].tolist()
    else:
        return arr[0], arr[1]


def replace_nan_with_value(
    x: OneDimArray,
    y: OneDimArray,
    value: Any,
) -> tuple[np.ndarray, np.ndarray]:
    
    x = np.array(
        [v if v == v and v is not None else value for v in x]
    )  # NaN != NaN
    y = np.array([v if v == v and v is not None else value for v in y])
    return x, y
