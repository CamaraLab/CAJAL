from _typeshed import Incomplete

copyright: Incomplete
version: Incomplete
BROADCAST_INTERVAL: int

class Discover:
    base: Incomplete
    hosts: Incomplete
    isclient: Incomplete
    def __init__(self, base, isclient: bool = ...) -> None: ...
    interface_addr: Incomplete
    broadcast_addr: Incomplete
    bsocket: Incomplete
    def run(self, interface_addr, broadcast_addr) -> None: ...
    def broadcast(self) -> None: ...
    socket: Incomplete
    def listen(self) -> None: ...
