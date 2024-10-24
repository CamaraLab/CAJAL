from ._sputils import isintlike as isintlike, isscalarlike as isscalarlike
from _typeshed import Incomplete

class spmatrix:
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __pow__(self, other): ...
    __dict__: Incomplete
    def set_shape(self, shape) -> None: ...
    def get_shape(self): ...
    shape: Incomplete
    def asfptype(self): ...
    def getmaxprint(self): ...
    def getformat(self): ...
    def getnnz(self, axis: Incomplete | None = ...): ...
    def getH(self): ...
    def getcol(self, j): ...
    def getrow(self, i): ...
