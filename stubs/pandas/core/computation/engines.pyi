import abc
from _typeshed import Incomplete
from pandas.core.computation.align import align_terms as align_terms, reconstruct_object as reconstruct_object
from pandas.core.computation.expr import Expr as Expr
from pandas.core.computation.ops import MATHOPS as MATHOPS, REDUCTIONS as REDUCTIONS
from pandas.errors import NumExprClobberingError as NumExprClobberingError

class AbstractEngine(metaclass=abc.ABCMeta):
    has_neg_frac: bool
    expr: Incomplete
    aligned_axes: Incomplete
    result_type: Incomplete
    def __init__(self, expr) -> None: ...
    def convert(self) -> str: ...
    def evaluate(self) -> object: ...

class NumExprEngine(AbstractEngine):
    has_neg_frac: bool

class PythonEngine(AbstractEngine):
    has_neg_frac: bool
    def evaluate(self): ...

ENGINES: dict[str, type[AbstractEngine]]
