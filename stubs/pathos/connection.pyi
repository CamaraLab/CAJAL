from _typeshed import Incomplete

class PipeException(Exception): ...

class Pipe:
    verbose: bool
    name: Incomplete
    background: Incomplete
    stdin: Incomplete
    codec: Incomplete
    message: Incomplete
    def __init__(self, name: Incomplete | None = ..., **kwds) -> None: ...
    def config(self, **kwds): ...
    def launch(self) -> None: ...
    def response(self): ...
    def pid(self): ...
    def kill(self) -> None: ...
    __call__: Incomplete