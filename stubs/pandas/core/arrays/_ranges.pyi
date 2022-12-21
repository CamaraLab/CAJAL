import numpy as np
from pandas._libs.lib import i8max as i8max
from pandas._libs.tslibs import BaseOffset as BaseOffset, OutOfBoundsDatetime as OutOfBoundsDatetime, Timedelta as Timedelta, Timestamp as Timestamp, iNaT as iNaT
from pandas._typing import npt as npt

def generate_regular_range(start: Union[Timestamp, Timedelta, None], end: Union[Timestamp, Timedelta, None], periods: Union[int, None], freq: BaseOffset) -> npt.NDArray[np.intp]: ...
