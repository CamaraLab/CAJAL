from _typeshed import Incomplete
from pandas._libs.tslibs import Timestamp as Timestamp
from pandas.compat.chainmap import DeepChainMap as DeepChainMap
from pandas.errors import UndefinedVariableError as UndefinedVariableError

def ensure_scope(level: int, global_dict: Incomplete | None = ..., local_dict: Incomplete | None = ..., resolvers=..., target: Incomplete | None = ..., **kwargs) -> Scope: ...

DEFAULT_GLOBALS: Incomplete

class Scope:
    level: int
    scope: DeepChainMap
    resolvers: DeepChainMap
    temps: dict
    target: Incomplete
    def __init__(self, level: int, global_dict: Incomplete | None = ..., local_dict: Incomplete | None = ..., resolvers=..., target: Incomplete | None = ...) -> None: ...
    @property
    def has_resolvers(self) -> bool: ...
    def resolve(self, key: str, is_local: bool): ...
    def swapkey(self, old_key: str, new_key: str, new_value: Incomplete | None = ...) -> None: ...
    def add_tmp(self, value) -> str: ...
    @property
    def ntemps(self) -> int: ...
    @property
    def full_scope(self) -> DeepChainMap: ...
