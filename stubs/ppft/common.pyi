import io
from _typeshed import Incomplete

long = int
file = io.IOBase

def str_(byte): ...
def b_(string): ...

copyright: str
parent: Incomplete

def start_thread(name, target, args=..., kwargs=..., daemon: bool = ...): ...
def get_class_hierarchy(clazz): ...
def is_not_imported(arg, modules): ...

class portnumber:
    min: Incomplete
    max: Incomplete
    first: int
    current: int
    def __init__(self, min: int = ..., max=...) -> None: ...
    def __call__(self): ...

def randomport(min: int = ..., max: int = ...): ...