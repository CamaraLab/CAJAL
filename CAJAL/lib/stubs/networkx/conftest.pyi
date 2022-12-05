from _typeshed import Incomplete

def pytest_addoption(parser) -> None: ...
def pytest_configure(config) -> None: ...
def pytest_collection_modifyitems(config, items) -> None: ...
def set_warnings() -> None: ...
def add_nx(doctest_namespace) -> None: ...

has_numpy: bool
has_scipy: bool
has_matplotlib: bool
has_pandas: bool
has_pygraphviz: bool
has_yaml: bool
has_pydot: bool
has_ogr: bool
has_sympy: bool
collect_ignore: Incomplete
needs_numpy: Incomplete
needs_scipy: Incomplete
needs_matplotlib: Incomplete
needs_pandas: Incomplete
needs_yaml: Incomplete
needs_pygraphviz: Incomplete
needs_pydot: Incomplete
needs_ogr: Incomplete
needs_sympy: Incomplete
