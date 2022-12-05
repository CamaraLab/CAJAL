from .test_digraph import BaseDiGraphTester as BaseDiGraphTester, TestDiGraph as _TestDiGraph
from .test_graph import BaseGraphTester as BaseGraphTester, TestGraph as _TestGraph
from .test_multidigraph import TestMultiDiGraph as _TestMultiDiGraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
from _typeshed import Incomplete

def test_factories() -> None: ...

class TestSpecialGraph(_TestGraph):
    Graph: Incomplete
    def setup_method(self) -> None: ...

class TestOrderedGraph(_TestGraph):
    Graph: Incomplete
    def setup_method(self) -> None: ...

class TestThinGraph(BaseGraphTester):
    Graph: Incomplete
    k3adj: Incomplete
    k3edges: Incomplete
    k3nodes: Incomplete
    K3: Incomplete
    def setup_method(self): ...

class TestSpecialDiGraph(_TestDiGraph):
    Graph: Incomplete
    def setup_method(self) -> None: ...

class TestOrderedDiGraph(_TestDiGraph):
    Graph: Incomplete
    def setup_method(self) -> None: ...

class TestThinDiGraph(BaseDiGraphTester):
    Graph: Incomplete
    k3adj: Incomplete
    k3edges: Incomplete
    k3nodes: Incomplete
    K3: Incomplete
    P3: Incomplete
    def setup_method(self): ...

class TestSpecialMultiGraph(_TestMultiGraph):
    Graph: Incomplete
    def setup_method(self) -> None: ...

class TestOrderedMultiGraph(_TestMultiGraph):
    Graph: Incomplete
    def setup_method(self) -> None: ...

class TestSpecialMultiDiGraph(_TestMultiDiGraph):
    Graph: Incomplete
    def setup_method(self) -> None: ...

class TestOrderedMultiDiGraph(_TestMultiDiGraph):
    Graph: Incomplete
    def setup_method(self) -> None: ...
