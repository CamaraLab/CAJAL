from collections.abc import Callable as Callable
from pandas import Series as Series
from pandas._typing import NpDtype as NpDtype, Scalar as Scalar
from pandas.core.dtypes.common import is_scalar as is_scalar
from pandas.core.dtypes.missing import isna as isna
from pandas.core.strings.base import BaseStringArrayMethods as BaseStringArrayMethods

class ObjectStringArrayMixin(BaseStringArrayMethods):
    def __len__(self) -> int: ...
