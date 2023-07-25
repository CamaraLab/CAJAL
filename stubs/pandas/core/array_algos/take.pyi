import numpy as np
from _typeshed import Incomplete
from pandas._libs import lib as lib
from pandas._typing import ArrayLike as ArrayLike, npt as npt
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray as NDArrayBackedExtensionArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike
from pandas.core.dtypes.cast import maybe_promote as maybe_promote
from pandas.core.dtypes.common import ensure_platform_int as ensure_platform_int, is_1d_only_ea_obj as is_1d_only_ea_obj
from pandas.core.dtypes.missing import na_value_for_dtype as na_value_for_dtype
from typing import overload

@overload
def take_nd(arr: np.ndarray, indexer, axis: int = ..., fill_value=..., allow_fill: bool = ...) -> np.ndarray: ...
@overload
def take_nd(arr: ExtensionArray, indexer, axis: int = ..., fill_value=..., allow_fill: bool = ...) -> ArrayLike: ...
def take_1d(arr: ArrayLike, indexer: npt.NDArray[np.intp], fill_value: Incomplete | None = ..., allow_fill: bool = ..., mask: Union[npt.NDArray[np.bool_], None] = ...) -> ArrayLike: ...
def take_2d_multi(arr: np.ndarray, indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]], fill_value=...) -> np.ndarray: ...
