from _typeshed import Incomplete
from scipy.sparse.linalg._interface import LinearOperator

SNAUPD_ERRORS = DNAUPD_ERRORS
CNAUPD_ERRORS = ZNAUPD_ERRORS
SSAUPD_ERRORS = DSAUPD_ERRORS

class ArpackError(RuntimeError):
    def __init__(self, info, infodict=...) -> None: ...

class ArpackNoConvergence(ArpackError):
    eigenvalues: Incomplete
    eigenvectors: Incomplete
    def __init__(self, msg, eigenvalues, eigenvectors) -> None: ...

class _ArpackParams:
    resid: Incomplete
    sigma: int
    v: Incomplete
    iparam: Incomplete
    mode: Incomplete
    n: Incomplete
    tol: Incomplete
    k: Incomplete
    maxiter: Incomplete
    ncv: Incomplete
    which: Incomplete
    tp: Incomplete
    info: Incomplete
    converged: bool
    ido: int
    def __init__(self, n, k, tp, mode: int = ..., sigma: Incomplete | None = ..., ncv: Incomplete | None = ..., v0: Incomplete | None = ..., maxiter: Incomplete | None = ..., which: str = ..., tol: int = ...) -> None: ...

class _SymmetricArpackParams(_ArpackParams):
    OP: Incomplete
    B: Incomplete
    bmat: str
    OPa: Incomplete
    OPb: Incomplete
    A_matvec: Incomplete
    workd: Incomplete
    workl: Incomplete
    iterate_infodict: Incomplete
    extract_infodict: Incomplete
    ipntr: Incomplete
    def __init__(self, n, k, tp, matvec, mode: int = ..., M_matvec: Incomplete | None = ..., Minv_matvec: Incomplete | None = ..., sigma: Incomplete | None = ..., ncv: Incomplete | None = ..., v0: Incomplete | None = ..., maxiter: Incomplete | None = ..., which: str = ..., tol: int = ...) -> None: ...
    converged: bool
    def iterate(self) -> None: ...
    def extract(self, return_eigenvectors): ...

class _UnsymmetricArpackParams(_ArpackParams):
    OP: Incomplete
    B: Incomplete
    bmat: str
    OPa: Incomplete
    OPb: Incomplete
    matvec: Incomplete
    workd: Incomplete
    workl: Incomplete
    iterate_infodict: Incomplete
    extract_infodict: Incomplete
    ipntr: Incomplete
    rwork: Incomplete
    def __init__(self, n, k, tp, matvec, mode: int = ..., M_matvec: Incomplete | None = ..., Minv_matvec: Incomplete | None = ..., sigma: Incomplete | None = ..., ncv: Incomplete | None = ..., v0: Incomplete | None = ..., maxiter: Incomplete | None = ..., which: str = ..., tol: int = ...) -> None: ...
    converged: bool
    def iterate(self) -> None: ...
    def extract(self, return_eigenvectors): ...

class SpLuInv(LinearOperator):
    M_lu: Incomplete
    shape: Incomplete
    dtype: Incomplete
    isreal: Incomplete
    def __init__(self, M) -> None: ...

class LuInv(LinearOperator):
    M_lu: Incomplete
    shape: Incomplete
    dtype: Incomplete
    def __init__(self, M) -> None: ...

class IterInv(LinearOperator):
    M: Incomplete
    dtype: Incomplete
    shape: Incomplete
    ifunc: Incomplete
    tol: Incomplete
    def __init__(self, M, ifunc=..., tol: int = ...) -> None: ...

class IterOpInv(LinearOperator):
    A: Incomplete
    M: Incomplete
    sigma: Incomplete
    OP: Incomplete
    shape: Incomplete
    ifunc: Incomplete
    tol: Incomplete
    def __init__(self, A, M, sigma, ifunc=..., tol: int = ...) -> None: ...
    @property
    def dtype(self): ...

def eigs(A, k: int = ..., M: Incomplete | None = ..., sigma: Incomplete | None = ..., which: str = ..., v0: Incomplete | None = ..., ncv: Incomplete | None = ..., maxiter: Incomplete | None = ..., tol: int = ..., return_eigenvectors: bool = ..., Minv: Incomplete | None = ..., OPinv: Incomplete | None = ..., OPpart: Incomplete | None = ...): ...
def eigsh(A, k: int = ..., M: Incomplete | None = ..., sigma: Incomplete | None = ..., which: str = ..., v0: Incomplete | None = ..., ncv: Incomplete | None = ..., maxiter: Incomplete | None = ..., tol: int = ..., return_eigenvectors: bool = ..., Minv: Incomplete | None = ..., OPinv: Incomplete | None = ..., mode: str = ...): ...
