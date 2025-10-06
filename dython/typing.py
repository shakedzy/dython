import numpy as np
import pandas as pd
from typing import Sequence, Any


Number = int | float
OneDimArray = Sequence[Number | str] | pd.Series | np.ndarray[Any, np.dtype[np.int64] | np.dtype[np.float64] | np.dtype[np.str_]]
TwoDimArray = np.ndarray[Any, np.dtype[np.int64] | np.dtype[np.float64] | np.dtype[np.str_]] | pd.DataFrame 
