from _typeshed import Incomplete
from typing import NamedTuple

def is_k_edge_connected(G, k): ...
def is_locally_k_edge_connected(G, s, t, k): ...
def k_edge_augmentation(G, k, avail: Incomplete | None = ..., weight: Incomplete | None = ..., partial: bool = ...) -> None: ...

class MetaEdge(NamedTuple):
    meta_uv: Incomplete
    uv: Incomplete
    w: Incomplete
