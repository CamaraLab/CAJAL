from scipy.sparse import csr_matrix as csr_matrix, hstack as hstack

def test_csr_rowslice() -> None: ...
def test_csr_getrow() -> None: ...
def test_csr_getcol() -> None: ...
def test_csr_empty_slices(matrix_input, axis, expected_shape) -> None: ...
def test_csr_bool_indexing() -> None: ...
def test_csr_hstack_int64() -> None: ...
