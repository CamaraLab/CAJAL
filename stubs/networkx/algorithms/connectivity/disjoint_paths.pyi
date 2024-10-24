from _typeshed import Incomplete
from collections.abc import Generator
from networkx.algorithms.flow import edmonds_karp

default_flow_func = edmonds_karp

def edge_disjoint_paths(G, s, t, flow_func: Incomplete | None = ..., cutoff: Incomplete | None = ..., auxiliary: Incomplete | None = ..., residual: Incomplete | None = ...) -> Generator[Incomplete, None, None]: ...
def node_disjoint_paths(G, s, t, flow_func: Incomplete | None = ..., cutoff: Incomplete | None = ..., auxiliary: Incomplete | None = ..., residual: Incomplete | None = ...) -> Generator[Incomplete, None, None]: ...
