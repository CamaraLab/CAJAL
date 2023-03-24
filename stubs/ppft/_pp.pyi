import types
from . import transport as pptransport
from _typeshed import Incomplete

user = types
copyright: Incomplete
version: Incomplete
RECONNECT_WAIT_TIME: int
SHOW_EXPECTED_EXCEPTIONS: bool

class _Task:
    lock: Incomplete
    tid: Incomplete
    server: Incomplete
    callback: Incomplete
    callbackargs: Incomplete
    group: Incomplete
    finished: bool
    unpickled: bool
    def __init__(self, server, tid, callback: Incomplete | None = ..., callbackargs=..., group: str = ...) -> None: ...
    sresult: Incomplete
    def finalize(self, sresult) -> None: ...
    def __call__(self, raw_result: bool = ...): ...
    def wait(self) -> None: ...

class _Worker:
    command: Incomplete
    restart_on_free: Incomplete
    pickle_proto: Incomplete
    def __init__(self, restart_on_free, pickle_proto) -> None: ...
    t: Incomplete
    pid: Incomplete
    is_free: bool
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...
    def free(self) -> None: ...

class _RWorker(pptransport.CSocketTransport):
    server: Incomplete
    persistent: Incomplete
    host: Incomplete
    port: Incomplete
    secret: Incomplete
    address: Incomplete
    id: Incomplete
    socket_timeout: Incomplete
    def __init__(self, host, port, secret, server, message, persistent, socket_timeout) -> None: ...
    def __del__(self) -> None: ...
    is_free: bool
    def connect(self, message: Incomplete | None = ...): ...

class _Statistics:
    ncpus: Incomplete
    time: float
    njobs: int
    rworker: Incomplete
    def __init__(self, ncpus, rworker: Incomplete | None = ...) -> None: ...

class Template:
    job_server: Incomplete
    func: Incomplete
    depfuncs: Incomplete
    modules: Incomplete
    callback: Incomplete
    callbackargs: Incomplete
    group: Incomplete
    globals: Incomplete
    def __init__(self, job_server, func, depfuncs=..., modules=..., callback: Incomplete | None = ..., callbackargs=..., group: str = ..., globals: Incomplete | None = ...) -> None: ...
    def submit(self, *args): ...

class Server:
    default_port: int
    default_secret: str
    logger: Incomplete
    autopp_list: Incomplete
    ppservers: Incomplete
    auto_ppservers: Incomplete
    socket_timeout: Incomplete
    secret: Incomplete
    def __init__(self, ncpus: str = ..., ppservers=..., secret: Incomplete | None = ..., restart: bool = ..., proto: int = ..., socket_timeout: int = ...) -> None: ...
    def submit(self, func, args=..., depfuncs=..., modules=..., callback: Incomplete | None = ..., callbackargs=..., group: str = ..., globals: Incomplete | None = ...): ...
    def wait(self, group: Incomplete | None = ...) -> None: ...
    def get_ncpus(self): ...
    def set_ncpus(self, ncpus: str = ...) -> None: ...
    def get_active_nodes(self): ...
    def get_stats(self): ...
    def print_stats(self) -> None: ...
    def insert(self, sfunc, sargs, task: Incomplete | None = ...): ...
    def connect1(self, host, port, persistent: bool = ...) -> None: ...
    def __del__(self) -> None: ...
    def destroy(self) -> None: ...

class DestroyedServerError(RuntimeError): ...
