import numpy as np
import re
from pandas._typing import ArrayLike as ArrayLike, Scalar as Scalar, npt as npt
from pandas.core.dtypes.common import is_datetimelike_v_numeric as is_datetimelike_v_numeric, is_numeric_v_string_like as is_numeric_v_string_like, is_re as is_re, is_re_compilable as is_re_compilable, is_scalar as is_scalar
from pandas.core.dtypes.missing import isna as isna
from typing import Any, Pattern

def should_use_regex(regex: bool, to_replace: Any) -> bool: ...
def compare_or_regex_search(a: ArrayLike, b: Union[Scalar, Pattern], regex: bool, mask: npt.NDArray[np.bool_]) -> Union[ArrayLike, bool]: ...
def replace_regex(values: ArrayLike, rx: re.Pattern, value, mask: Union[npt.NDArray[np.bool_], None]) -> None: ...
