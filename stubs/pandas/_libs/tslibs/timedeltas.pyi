from typing import Any, ClassVar

from typing import overload
import datetime
_no_input: object
find_stack_level: function

class Components(tuple):
    _asdict: ClassVar[function] = ...
    _field_defaults: ClassVar[dict] = ...
    _fields: ClassVar[tuple] = ...
    _replace: ClassVar[function] = ...
    __getnewargs__: ClassVar[function] = ...
    __match_args__: ClassVar[tuple] = ...
    __slots__: ClassVar[tuple] = ...
    days: Any
    hours: Any
    microseconds: Any
    milliseconds: Any
    minutes: Any
    nanoseconds: Any
    seconds: Any
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def _make(cls, *args, **kwargs) -> Any: ...

class MinMaxReso:
    def __init__(self, *args, **kwargs) -> None: ...
    def __get__(self, instance, owner) -> Any: ...
    def __set__(self, instance, value) -> Any: ...

class OutOfBoundsDatetime(ValueError): ...

class OutOfBoundsTimedelta(ValueError): ...

class RoundTo:
    @property
    def MINUS_INFTY(self) -> Any: ...
    @property
    def NEAREST_HALF_EVEN(self) -> Any: ...
    @property
    def NEAREST_HALF_MINUS_INFTY(self) -> Any: ...
    @property
    def NEAREST_HALF_PLUS_INFTY(self) -> Any: ...
    @property
    def PLUS_INFTY(self) -> Any: ...

class Timedelta(_Timedelta):
    _req_any_kwargs_new: ClassVar[set] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _round(self, *args, **kwargs) -> Any: ...
    def ceil(self, *args, **kwargs) -> Any: ...
    def floor(self, *args, **kwargs) -> Any: ...
    def round(self, *args, **kwargs) -> Any: ...
    def __abs__(self) -> Any: ...
    def __add__(self, other) -> Any: ...
    def __divmod__(self, other) -> Any: ...
    def __floordiv__(self, other) -> Any: ...
    def __mod__(self, other) -> Any: ...
    def __mul__(self, other) -> Any: ...
    def __neg__(self) -> Any: ...
    def __pos__(self) -> Any: ...
    def __radd__(self, other) -> Any: ...
    def __rdivmod__(self, other) -> Any: ...
    def __reduce__(self) -> Any: ...
    def __rfloordiv__(self, other) -> Any: ...
    def __rmod__(self, other) -> Any: ...
    def __rmul__(self, other) -> Any: ...
    def __rsub__(self, other) -> Any: ...
    def __rtruediv__(self, other) -> Any: ...
    def __setstate__(self, state) -> Any: ...
    def __sub__(self, other) -> Any: ...
    def __truediv__(self, other) -> Any: ...

class _Timedelta(datetime.timedelta):
    __array_priority__: ClassVar[int] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _d: Any
    _h: Any
    _is_populated: Any
    _m: Any
    _ms: Any
    _ns: Any
    _reso: Any
    _s: Any
    _us: Any
    asm8: Any
    components: Any
    days: Any
    delta: Any
    freq: Any
    is_populated: Any
    max: Any
    microseconds: Any
    min: Any
    nanoseconds: Any
    resolution: Any
    resolution_string: Any
    seconds: Any
    value: Any
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def _as_unit(self, *args, **kwargs) -> Any: ...
    @classmethod
    def _from_value_and_reso(cls, *args, **kwargs) -> Any: ...
    def _repr_base(self, *args, **kwargs) -> Any: ...
    @overload
    def isoformat(self) -> Any: ...
    @overload
    def isoformat(self) -> Any: ...
    @overload
    def isoformat(self) -> Any: ...
    def to_numpy(self, *args, **kwargs) -> Any: ...
    def to_pytimedelta(self) -> Any: ...
    def to_timedelta64(self, *args, **kwargs) -> Any: ...
    def total_seconds(self, *args, **kwargs) -> Any: ...
    def view(self, *args, **kwargs) -> Any: ...
    def __bool__(self) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __hash__(self) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __reduce_cython__(self, *args, **kwargs) -> Any: ...
    def __setstate_cython__(self, *args, **kwargs) -> Any: ...

def __pyx_unpickle_Enum(*args, **kwargs) -> Any: ...
def __pyx_unpickle__Timedelta(*args, **kwargs) -> Any: ...
def _binary_op_method_timedeltalike(*args, **kwargs) -> Any: ...
def _op_unary_method(*args, **kwargs) -> Any: ...
def _timedelta_unpickle(*args, **kwargs) -> Any: ...
def array_to_timedelta64(*args, **kwargs) -> Any: ...
def delta_to_nanoseconds(*args, **kwargs) -> Any: ...
def ints_to_pytimedelta(*args, **kwargs) -> Any: ...
def parse_timedelta_unit(*args, **kwargs) -> Any: ...
def round_nsint64(*args, **kwargs) -> Any: ...