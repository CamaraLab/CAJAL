import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing as is_monotonic_increasing

def add_mean(val: float, nobs: int, sum_x: float, neg_ct: int, compensation: float, num_consecutive_same_value: int, prev_value: float) -> tuple[int, float, int, float, int, float]: ...
def remove_mean(val: float, nobs: int, sum_x: float, neg_ct: int, compensation: float) -> tuple[int, float, int, float]: ...
def sliding_mean(values: np.ndarray, start: np.ndarray, end: np.ndarray, min_periods: int) -> np.ndarray: ...