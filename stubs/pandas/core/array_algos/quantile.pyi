import numpy as np
from pandas._typing import ArrayLike as ArrayLike, Scalar as Scalar, npt as npt
from pandas.compat.numpy import np_percentile_argname as np_percentile_argname
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype

def quantile_compat(values: ArrayLike, qs: npt.NDArray[np.float64], interpolation: str) -> ArrayLike: ...
def quantile_with_mask(values: np.ndarray, mask: npt.NDArray[np.bool_], fill_value, qs: npt.NDArray[np.float64], interpolation: str) -> np.ndarray: ...
