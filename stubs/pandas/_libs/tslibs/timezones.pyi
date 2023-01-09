from typing import Any, ClassVar

from typing import overload
import datetime
import dateutil.tz._common
import dateutil.tz.tz
# import dateutil.tz.tz.__get_gettz.<locals>
import pytz
UTC: pytz.UTC
# dateutil_gettz: dateutil.tz.tz.__get_gettz.<locals>.GettzFunc
dst_cache: dict
import_optional_dependency: function

class ZoneInfo(datetime.tzinfo):
    key: Any
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    @classmethod
    def _unpickle(cls, *args, **kwargs) -> Any: ...
    @classmethod
    def clear_cache(cls, *args, **kwargs) -> Any: ...
    def dst(self, *args, **kwargs) -> Any: ...
    @classmethod
    def from_file(cls, *args, **kwargs) -> Any: ...
    def fromutc(self, *args, **kwargs) -> Any: ...
    @classmethod
    def no_cache(cls, *args, **kwargs) -> Any: ...
    def tzname(self, *args, **kwargs) -> Any: ...
    def utcoffset(self, *args, **kwargs) -> Any: ...
    @classmethod
    def __init_subclass__(cls, *args, **kwargs) -> Any: ...
    def __reduce__(self) -> Any: ...

class _dateutil_tzfile(dateutil.tz._common._tzinfo):
    __init__: ClassVar[function] = ...
    _find_last_transition: ClassVar[function] = ...
    _find_ttinfo: ClassVar[function] = ...
    _get_ttinfo: ClassVar[function] = ...
    _read_tzfile: ClassVar[function] = ...
    _resolve_ambiguous_time: ClassVar[function] = ...
    _set_tzdata: ClassVar[function] = ...
    dst: ClassVar[function] = ...
    fromutc: ClassVar[function] = ...
    is_ambiguous: ClassVar[function] = ...
    tzname: ClassVar[function] = ...
    utcoffset: ClassVar[function] = ...
    __eq__: ClassVar[function] = ...
    __hash__: ClassVar[None] = ...
    __ne__: ClassVar[function] = ...
    __reduce__: ClassVar[function] = ...
    __reduce_ex__: ClassVar[function] = ...

class _dateutil_tzlocal(dateutil.tz._common._tzinfo):
    __init__: ClassVar[function] = ...
    _isdst: ClassVar[function] = ...
    _naive_is_dst: ClassVar[function] = ...
    dst: ClassVar[function] = ...
    is_ambiguous: ClassVar[function] = ...
    tzname: ClassVar[function] = ...
    utcoffset: ClassVar[function] = ...
    __eq__: ClassVar[function] = ...
    __hash__: ClassVar[None] = ...
    __ne__: ClassVar[function] = ...
    def __reduce__(self) -> Any: ...

class _dateutil_tzutc(datetime.tzinfo):
    _TzSingleton__instance: ClassVar[dateutil.tz.tz.tzutc] = ...
    dst: ClassVar[function] = ...
    fromutc: ClassVar[function] = ...
    is_ambiguous: ClassVar[function] = ...
    tzname: ClassVar[function] = ...
    utcoffset: ClassVar[function] = ...
    __eq__: ClassVar[function] = ...
    __hash__: ClassVar[None] = ...
    __ne__: ClassVar[function] = ...
    def __reduce__(self) -> Any: ...

class _pytz_BaseTzInfo(datetime.tzinfo):
    _tzname: ClassVar[None] = ...
    _utcoffset: ClassVar[None] = ...
    zone: ClassVar[None] = ...

class timezone(datetime.tzinfo):
    max: ClassVar[datetime.timezone] = ...
    min: ClassVar[datetime.timezone] = ...
    utc: ClassVar[datetime.timezone] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def dst(self, *args, **kwargs) -> Any: ...
    def fromutc(self, *args, **kwargs) -> Any: ...
    def tzname(self, *args, **kwargs) -> Any: ...
    def utcoffset(self, *args, **kwargs) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __getinitargs__(self) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __hash__(self) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...

def __pyx_unpickle_Enum(*args, **kwargs) -> Any: ...
def _p_tz_cache_key(*args, **kwargs) -> Any: ...
def get_timezone(*args, **kwargs) -> Any: ...
def infer_tzinfo(*args, **kwargs) -> Any: ...
def is_fixed_offset(*args, **kwargs) -> Any: ...
def is_utc(*args, **kwargs) -> Any: ...
def maybe_get_tz(*args, **kwargs) -> Any: ...
def tz_compare(*args, **kwargs) -> Any: ...
@overload
def tz_standardize(tz) -> Any: ...
@overload
def tz_standardize(tz) -> Any: ...