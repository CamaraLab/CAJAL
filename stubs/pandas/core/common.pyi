import numpy as np
from _typeshed import Incomplete
from collections.abc import Generator
from pandas import Index as Index
from pandas._libs import lib as lib
from pandas._typing import AnyArrayLike as AnyArrayLike, ArrayLike as ArrayLike, NpDtype as NpDtype, RandomState as RandomState, T as T
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike as construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_array_like as is_array_like, is_bool_dtype as is_bool_dtype, is_extension_array_dtype as is_extension_array_dtype, is_integer as is_integer
from pandas.core.dtypes.generic import ABCExtensionArray as ABCExtensionArray, ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.inference import iterable_not_string as iterable_not_string
from pandas.core.dtypes.missing import isna as isna
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, Callable, Collection, Hashable, Iterable, Iterator, Sequence, overload

def flatten(line) -> Generator[Incomplete, None, None]: ...
def consensus_name_attr(objs): ...
def is_bool_indexer(key: Any) -> bool: ...
def cast_scalar_indexer(val, warn_float: bool = ...): ...
def not_none(*args): ...
def any_none(*args) -> bool: ...
def all_none(*args) -> bool: ...
def any_not_none(*args) -> bool: ...
def all_not_none(*args) -> bool: ...
def count_not_none(*args) -> int: ...
@overload
def asarray_tuplesafe(values: Union[ArrayLike, list, tuple, zip], dtype: Union[NpDtype, None] = ...) -> np.ndarray: ...
@overload
def asarray_tuplesafe(values: Iterable, dtype: Union[NpDtype, None] = ...) -> ArrayLike: ...
def index_labels_to_array(labels: Union[np.ndarray, Iterable], dtype: Union[NpDtype, None] = ...) -> np.ndarray: ...
def maybe_make_list(obj): ...
def maybe_iterable_to_list(obj: Union[Iterable[T], T]) -> Union[Collection[T], T]: ...
def is_null_slice(obj) -> bool: ...
def is_true_slices(line) -> list[bool]: ...
def is_full_slice(obj, line: int) -> bool: ...
def get_callable_name(obj): ...
def apply_if_callable(maybe_callable, obj, **kwargs): ...
def standardize_mapping(into): ...
@overload
def random_state(state: np.random.Generator) -> np.random.Generator: ...
@overload
def random_state(state: Union[int, ArrayLike, np.random.BitGenerator, np.random.RandomState, None]) -> np.random.RandomState: ...
def pipe(obj, func: Union[Callable[..., T], tuple[Callable[..., T], str]], *args, **kwargs) -> T: ...
def get_rename_function(mapper): ...
def convert_to_list_like(values: Union[Hashable, Iterable, AnyArrayLike]) -> Union[list, AnyArrayLike]: ...
def temp_setattr(obj, attr: str, value) -> Iterator[None]: ...
def require_length_match(data, index: Index) -> None: ...
def get_cython_func(arg: Callable) -> Union[str, None]: ...
def is_builtin_func(arg): ...
def fill_missing_names(names: Sequence[Union[Hashable, None]]) -> list[Hashable]: ...
def resolve_numeric_only(numeric_only: Union[bool, None, lib.NoDefault]) -> bool: ...
def deprecate_numeric_only_default(cls, name: str, deprecate_none: bool = ...) -> None: ...
