from _typeshed import Incomplete
from pandas._config import get_option as get_option
from pandas._typing import F as F
from pandas.compat import IS64 as IS64, is_platform_windows as is_platform_windows
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.computation.expressions import NUMEXPR_INSTALLED as NUMEXPR_INSTALLED, USE_NUMEXPR as USE_NUMEXPR
from pandas.util.version import Version as Version
from typing import Callable, Iterator

def safe_import(mod_name: str, min_version: Union[str, None] = ...): ...
def skip_if_installed(package: str): ...
def skip_if_no(package: str, min_version: Union[str, None] = ...): ...

skip_if_no_mpl: Incomplete
skip_if_mpl: Incomplete
skip_if_32bit: Incomplete
skip_if_windows: Incomplete
skip_if_not_us_locale: Incomplete
skip_if_no_scipy: Incomplete
skip_if_no_ne: Incomplete

def skip_if_np_lt(ver_str: str, *args, reason: Union[str, None] = ...): ...
def parametrize_fixture_doc(*args) -> Callable[[F], F]: ...
def check_file_leaks(func) -> Callable: ...
def file_leak_context() -> Iterator[None]: ...
def async_mark(): ...
def mark_array_manager_not_yet_implemented(request) -> None: ...

skip_array_manager_not_yet_implemented: Incomplete
skip_array_manager_invalid_test: Incomplete
