from _typeshed import Incomplete
from collections.abc import Generator
from enum import Enum
from typing import Any, NamedTuple

def read_gml(path, label: str = ..., destringizer: Incomplete | None = ...): ...
def parse_gml(lines, label: str = ..., destringizer: Incomplete | None = ...): ...

class Pattern(Enum):
    KEYS: int
    REALS: int
    INTS: int
    STRINGS: int
    DICT_START: int
    DICT_END: int
    COMMENT_WHITESPACE: int

class Token(NamedTuple):
    category: Pattern
    value: Any
    line: int
    position: int

def generate_gml(G, stringizer: Incomplete | None = ...) -> Generator[Incomplete, None, None]: ...
def write_gml(G, path, stringizer: Incomplete | None = ...) -> None: ...
