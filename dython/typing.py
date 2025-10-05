import numpy as np
import pandas as pd
from typing import Sequence


Number = int | float
OneDimArray = Sequence[Number] | Sequence[str] | pd.Series | np.ndarray
TwoDimArray = np.ndarray | pd.DataFrame 
