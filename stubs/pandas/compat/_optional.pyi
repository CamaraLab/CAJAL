import types
from _typeshed import Incomplete
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util.version import Version as Version

VERSIONS: Incomplete
INSTALL_MAPPING: Incomplete

def get_version(module: types.ModuleType) -> str: ...
def import_optional_dependency(name: str, extra: str = ..., errors: str = ..., min_version: Union[str, None] = ...): ...
