import numpy as np
from _typeshed import Incomplete
from datetime import datetime, tzinfo
from pandas import Interval as Interval
from pandas._libs import NaTType as NaTType, Period as Period, Timedelta as Timedelta, Timestamp as Timestamp
from pandas._libs.tslibs import BaseOffset as BaseOffset
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype as ExtensionDtype
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby.generic import DataFrameGroupBy as DataFrameGroupBy, GroupBy as GroupBy, SeriesGroupBy as SeriesGroupBy
from pandas.core.indexes.base import Index as Index
from pandas.core.internals import ArrayManager as ArrayManager, BlockManager as BlockManager, SingleArrayManager as SingleArrayManager, SingleBlockManager as SingleBlockManager
from pandas.core.resample import Resampler as Resampler
from pandas.core.series import Series as Series
from pandas.core.window.rolling import BaseWindow as BaseWindow
from pandas.io.formats.format import EngFormatter as EngFormatter
from typing import Any, Callable, Dict, Hashable, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, Type as type_t, TypeVar, Union

ScalarLike_co = Union[int, float, complex, str, bytes, np.generic]
NumpyValueArrayLike: Incomplete
NumpySorter: Incomplete
HashableT = TypeVar('HashableT', bound=Hashable)
ArrayLike: Incomplete
AnyArrayLike: Incomplete
PythonScalar = Union[str, float, bool]
DatetimeLikeScalar: Incomplete
PandasScalar: Incomplete
Scalar = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, datetime]
IntStrT = TypeVar('IntStrT', int, str)
TimestampConvertibleTypes: Incomplete
TimedeltaConvertibleTypes: Incomplete
Timezone = Union[str, tzinfo]
NDFrameT = TypeVar('NDFrameT', bound='NDFrame')
NumpyIndexT = TypeVar('NumpyIndexT', np.ndarray, 'Index')
Axis = Union[str, int]
IndexLabel = Union[Hashable, Sequence[Hashable]]
Level = Hashable
Shape = Tuple[int, ...]
Suffixes = Tuple[Optional[str], Optional[str]]
Ordered = Optional[bool]
JSONSerializable = Optional[Union[PythonScalar, List, Dict]]
Frequency: Incomplete
Axes = Union[AnyArrayLike, List, range]
RandomState: Incomplete
NpDtype = Union[str, np.dtype, type_t[Union[str, complex, bool, object]]]
Dtype: Incomplete
AstypeArg: Incomplete
DtypeArg = Union[Dtype, Dict[Hashable, Dtype]]
DtypeObj: Incomplete
ConvertersArg = Dict[Hashable, Callable[[Dtype], Dtype]]
ParseDatesArg = Union[bool, List[Hashable], List[List[Hashable]], Dict[Hashable, List[Hashable]]]
Renamer = Union[Mapping[Any, Hashable], Callable[[Any], Hashable]]
T = TypeVar('T')
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)
ValueKeyFunc: Incomplete
IndexKeyFunc: Incomplete
AggFuncTypeBase = Union[Callable, str]
AggFuncTypeDict = Dict[Hashable, Union[AggFuncTypeBase, List[AggFuncTypeBase]]]
AggFuncType = Union[AggFuncTypeBase, List[AggFuncTypeBase], AggFuncTypeDict]
AggObjType: Incomplete
PythonFuncType = Callable[[Any], Any]
AnyStr_cov = TypeVar('AnyStr_cov', str, bytes, covariant=True)
AnyStr_con = TypeVar('AnyStr_con', str, bytes, contravariant=True)

class BaseBuffer(Protocol):
    @property
    def mode(self) -> str: ...
    def fileno(self) -> int: ...
    def seek(self, __offset: int, __whence: int = ...) -> int: ...
    def seekable(self) -> bool: ...
    def tell(self) -> int: ...

class ReadBuffer(BaseBuffer, Protocol[AnyStr_cov]):
    def read(self, __n: Union[int, None] = ...) -> AnyStr_cov: ...

class WriteBuffer(BaseBuffer, Protocol[AnyStr_con]):
    def write(self, __b: AnyStr_con) -> Any: ...
    def flush(self) -> Any: ...

class ReadPickleBuffer(ReadBuffer[bytes], Protocol):
    def readline(self) -> AnyStr_cov: ...

class WriteExcelBuffer(WriteBuffer[bytes], Protocol):
    def truncate(self, size: Union[int, None] = ...) -> int: ...

class ReadCsvBuffer(ReadBuffer[AnyStr_cov], Protocol[AnyStr_cov]):
    def __iter__(self) -> Iterator[AnyStr_cov]: ...
    def readline(self) -> AnyStr_cov: ...
    @property
    def closed(self) -> bool: ...

FilePath: Incomplete
StorageOptions = Optional[Dict[str, Any]]
CompressionDict = Dict[str, Any]
CompressionOptions: Incomplete
FormattersType = Union[List[Callable], Tuple[Callable, ...], Mapping[Union[str, int], Callable]]
ColspaceType = Mapping[Hashable, Union[str, int]]
FloatFormatType: Incomplete
ColspaceArgType = Union[str, int, Sequence[Union[str, int]], Mapping[Hashable, Union[str, int]]]
FillnaOptions: Incomplete
Manager: Incomplete
SingleManager: Incomplete
Manager2D: Incomplete
ScalarIndexer = Union[int, np.integer]
SequenceIndexer = Union[slice, List[int], np.ndarray]
PositionalIndexer = Union[ScalarIndexer, SequenceIndexer]
PositionalIndexerTuple = Tuple[PositionalIndexer, PositionalIndexer]
PositionalIndexer2D = Union[PositionalIndexer, PositionalIndexerTuple]
TakeIndexer: Incomplete
IgnoreRaise: Incomplete
WindowingRankType: Incomplete
CSVEngine: Incomplete
XMLParsers: Incomplete
IntervalLeftRight: Incomplete
IntervalClosedType: Incomplete
DatetimeNaTType: Incomplete
DateTimeErrorChoices: Incomplete
SortKind: Incomplete
NaPosition: Incomplete
QuantileInterpolation: Incomplete
PlottingOrientation: Incomplete
