from pandas._typing import Scalar as Scalar
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from typing import Callable

def generate_shared_aggregator(func: Callable[..., Scalar], nopython: bool, nogil: bool, parallel: bool): ...
