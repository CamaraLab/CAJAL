from _typeshed import Incomplete
from pandas._libs import lib as lib
from pandas._typing import npt as npt
from pandas.core.arrays.numeric import NumericArray as NumericArray
from pandas.core.dtypes.cast import maybe_downcast_numeric as maybe_downcast_numeric
from pandas.core.dtypes.common import ensure_object as ensure_object, is_datetime_or_timedelta_dtype as is_datetime_or_timedelta_dtype, is_decimal as is_decimal, is_integer_dtype as is_integer_dtype, is_number as is_number, is_numeric_dtype as is_numeric_dtype, is_scalar as is_scalar, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCSeries as ABCSeries

def to_numeric(arg, errors: str = ..., downcast: Incomplete | None = ...): ...
