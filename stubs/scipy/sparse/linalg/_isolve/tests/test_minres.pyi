from .test_iterative import assert_normclose as assert_normclose
from scipy.sparse.linalg._isolve import minres as minres

def get_sample_problem(): ...
def test_singular() -> None: ...
def test_x0_is_used_by() -> None: ...
def test_shift() -> None: ...
def test_asymmetric_fail() -> None: ...
def test_minres_non_default_x0() -> None: ...
def test_minres_precond_non_default_x0() -> None: ...
def test_minres_precond_exact_x0() -> None: ...
