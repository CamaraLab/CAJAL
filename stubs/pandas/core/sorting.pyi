import numpy as np
from _typeshed import Incomplete
from pandas import MultiIndex as MultiIndex
from pandas._libs import algos as algos, hashtable as hashtable, lib as lib
from pandas._libs.hashtable import unique_label_indices as unique_label_indices
from pandas._typing import IndexKeyFunc as IndexKeyFunc, Level as Level, NaPosition as NaPosition, Shape as Shape, SortKind as SortKind, npt as npt
from pandas.core.arrays import ExtensionArray as ExtensionArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import ensure_int64 as ensure_int64, ensure_platform_int as ensure_platform_int, is_extension_array_dtype as is_extension_array_dtype
from pandas.core.dtypes.generic import ABCMultiIndex as ABCMultiIndex, ABCRangeIndex as ABCRangeIndex
from pandas.core.dtypes.missing import isna as isna
from pandas.core.indexes.base import Index as Index
from typing import Callable, Hashable, Iterable, Sequence

def get_indexer_indexer(target: Index, level: Union[Level, list[Level], None], ascending: Union[Sequence[bool], bool], kind: SortKind, na_position: NaPosition, sort_remaining: bool, key: IndexKeyFunc) -> Union[npt.NDArray[np.intp], None]: ...
def get_group_index(labels, shape: Shape, sort: bool, xnull: bool) -> npt.NDArray[np.int64]: ...
def get_compressed_ids(labels, sizes: Shape) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]: ...
def is_int64_overflow_possible(shape: Shape) -> bool: ...
def decons_obs_group_ids(comp_ids: npt.NDArray[np.intp], obs_ids: npt.NDArray[np.intp], shape: Shape, labels: Sequence[npt.NDArray[np.signedinteger]], xnull: bool) -> list[npt.NDArray[np.intp]]: ...
def indexer_from_factorized(labels, shape: Shape, compress: bool = ...) -> npt.NDArray[np.intp]: ...
def lexsort_indexer(keys, orders: Incomplete | None = ..., na_position: str = ..., key: Union[Callable, None] = ...) -> npt.NDArray[np.intp]: ...
def nargsort(items, kind: str = ..., ascending: bool = ..., na_position: str = ..., key: Union[Callable, None] = ..., mask: Union[npt.NDArray[np.bool_], None] = ...) -> npt.NDArray[np.intp]: ...
def nargminmax(values: ExtensionArray, method: str, axis: int = ...): ...
def ensure_key_mapped(values, key: Union[Callable, None], levels: Incomplete | None = ...): ...
def get_flattened_list(comp_ids: npt.NDArray[np.intp], ngroups: int, levels: Iterable[Index], labels: Iterable[np.ndarray]) -> list[tuple]: ...
def get_indexer_dict(label_list: list[np.ndarray], keys: list[Index]) -> dict[Hashable, npt.NDArray[np.intp]]: ...
def get_group_index_sorter(group_index: npt.NDArray[np.intp], ngroups: Union[int, None] = ...) -> npt.NDArray[np.intp]: ...
def compress_group_index(group_index: npt.NDArray[np.int64], sort: bool = ...) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...