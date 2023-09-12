from ._sputils import isintlike as isintlike
from _typeshed import Incomplete

INT_TYPES: Incomplete

class IndexMixin:
    def __getitem__(self, key): ...
    def __setitem__(self, key, x) -> None: ...
