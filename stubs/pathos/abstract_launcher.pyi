class AbstractPipeConnection:
    def __init__(self, *args, **kwds) -> None: ...

class AbstractWorkerPool:
    def __init__(self, *args, **kwds) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...
    def clear(self) -> None: ...
    def map(self, f, *args, **kwds) -> None: ...
    def imap(self, f, *args, **kwds) -> None: ...
    def uimap(self, f, *args, **kwds) -> None: ...
    def amap(self, f, *args, **kwds) -> None: ...
    def pipe(self, f, *args, **kwds) -> None: ...
    def apipe(self, f, *args, **kwds) -> None: ...