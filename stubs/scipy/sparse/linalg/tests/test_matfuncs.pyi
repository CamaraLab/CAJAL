from scipy.linalg import logm as logm
from scipy.sparse import SparseEfficiencyWarning as SparseEfficiencyWarning, csc_matrix as csc_matrix
from scipy.sparse._sputils import matrix as matrix
from scipy.sparse.linalg._matfuncs import MatrixPowerOperator as MatrixPowerOperator, ProductOperator as ProductOperator, expm as expm
from scipy.special import binom as binom, factorial as factorial

def test_onenorm_matrix_power_nnm() -> None: ...

class TestExpM:
    def test_zero_ndarray(self) -> None: ...
    def test_zero_sparse(self) -> None: ...
    def test_zero_matrix(self) -> None: ...
    def test_misc_types(self) -> None: ...
    def test_bidiagonal_sparse(self) -> None: ...
    def test_padecases_dtype_float(self) -> None: ...
    def test_padecases_dtype_complex(self) -> None: ...
    def test_padecases_dtype_sparse_float(self) -> None: ...
    def test_padecases_dtype_sparse_complex(self) -> None: ...
    def test_logm_consistency(self) -> None: ...
    def test_integer_matrix(self) -> None: ...
    def test_integer_matrix_2(self) -> None: ...
    def test_triangularity_perturbation(self) -> None: ...
    def test_burkardt_1(self) -> None: ...
    def test_burkardt_2(self) -> None: ...
    def test_burkardt_3(self) -> None: ...
    def test_burkardt_4(self) -> None: ...
    def test_burkardt_5(self) -> None: ...
    def test_burkardt_6(self) -> None: ...
    def test_burkardt_7(self) -> None: ...
    def test_burkardt_8(self) -> None: ...
    def test_burkardt_9(self) -> None: ...
    def test_burkardt_10(self) -> None: ...
    def test_burkardt_11(self) -> None: ...
    def test_burkardt_12(self) -> None: ...
    def test_burkardt_13(self) -> None: ...
    def test_burkardt_14(self) -> None: ...
    def test_pascal(self) -> None: ...
    def test_matrix_input(self) -> None: ...
    def test_exp_sinch_overflow(self) -> None: ...

class TestOperators:
    def test_product_operator(self) -> None: ...
    def test_matrix_power_operator(self) -> None: ...