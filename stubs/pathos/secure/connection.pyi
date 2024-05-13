from _typeshed import Incomplete
from pathos.connection import Pipe as _Pipe

class Pipe(_Pipe):
    launcher: Incomplete
    options: Incomplete
    host: Incomplete
    def __init__(self, name: Incomplete | None = ..., **kwds) -> None: ...
    message: Incomplete
    stdin: Incomplete
    background: Incomplete
    codec: Incomplete
    def config(self, **kwds): ...
    __call__: Incomplete
