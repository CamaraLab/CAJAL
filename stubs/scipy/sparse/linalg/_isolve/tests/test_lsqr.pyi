from _typeshed import Incomplete
from scipy.sparse.linalg import lsqr as lsqr

n: int
G: Incomplete
normal: Incomplete
norm: Incomplete
gg: Incomplete
hh: Incomplete
b: Incomplete
tol: float
atol_test: float
rtol_test: float
show: bool
maxit: Incomplete

def test_lsqr_basic() -> None: ...
def test_gh_2466() -> None: ...
def test_well_conditioned_problems() -> None: ...
def test_b_shapes() -> None: ...
def test_initialization() -> None: ...
