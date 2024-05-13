from .digraph import DiGraph
from .graph import Graph
from .multidigraph import MultiDiGraph
from .multigraph import MultiGraph
from _typeshed import Incomplete

class OrderedGraph(Graph):
    node_dict_factory: Incomplete
    adjlist_outer_dict_factory: Incomplete
    adjlist_inner_dict_factory: Incomplete
    edge_attr_dict_factory: Incomplete
    def __init__(self, incoming_graph_data: Incomplete | None = ..., **attr) -> None: ...

class OrderedDiGraph(DiGraph):
    node_dict_factory: Incomplete
    adjlist_outer_dict_factory: Incomplete
    adjlist_inner_dict_factory: Incomplete
    edge_attr_dict_factory: Incomplete
    def __init__(self, incoming_graph_data: Incomplete | None = ..., **attr) -> None: ...

class OrderedMultiGraph(MultiGraph):
    node_dict_factory: Incomplete
    adjlist_outer_dict_factory: Incomplete
    adjlist_inner_dict_factory: Incomplete
    edge_key_dict_factory: Incomplete
    edge_attr_dict_factory: Incomplete
    def __init__(self, incoming_graph_data: Incomplete | None = ..., **attr) -> None: ...

class OrderedMultiDiGraph(MultiDiGraph):
    node_dict_factory: Incomplete
    adjlist_outer_dict_factory: Incomplete
    adjlist_inner_dict_factory: Incomplete
    edge_key_dict_factory: Incomplete
    edge_attr_dict_factory: Incomplete
    def __init__(self, incoming_graph_data: Incomplete | None = ..., **attr) -> None: ...
