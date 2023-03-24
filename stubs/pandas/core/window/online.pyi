from _typeshed import Incomplete
from pandas.compat._optional import import_optional_dependency as import_optional_dependency

def generate_online_numba_ewma_func(nopython: bool, nogil: bool, parallel: bool): ...

class EWMMeanState:
    axis: Incomplete
    shape: Incomplete
    adjust: Incomplete
    ignore_na: Incomplete
    new_wt: Incomplete
    old_wt_factor: Incomplete
    old_wt: Incomplete
    last_ewm: Incomplete
    def __init__(self, com, adjust, ignore_na, axis, shape) -> None: ...
    def run_ewm(self, weighted_avg, deltas, min_periods, ewm_func): ...
    def reset(self) -> None: ...
