from _typeshed import Incomplete
from pathos.connection import Pipe as _Pipe

class FileNotFound(Exception): ...

class Copier(_Pipe):
    launcher: Incomplete
    options: Incomplete
    source: Incomplete
    destination: Incomplete
    def __init__(self, name: Incomplete | None = ..., **kwds) -> None: ...
    stdin: Incomplete
    background: Incomplete
    codec: Incomplete
    message: Incomplete
    def config(self, **kwds): ...
    __call__: Incomplete
