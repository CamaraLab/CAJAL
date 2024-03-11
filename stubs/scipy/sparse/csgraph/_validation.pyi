from ._tools import csgraph_from_dense as csgraph_from_dense, csgraph_from_masked as csgraph_from_masked, csgraph_masked_from_dense as csgraph_masked_from_dense, csgraph_to_dense as csgraph_to_dense
from _typeshed import Incomplete
from scipy.sparse import csr_matrix as csr_matrix, issparse as issparse

DTYPE: Incomplete

def validate_graph(csgraph, directed, dtype=..., csr_output: bool = ..., dense_output: bool = ..., copy_if_dense: bool = ..., copy_if_sparse: bool = ..., null_value_in: int = ..., null_value_out=..., infinity_null: bool = ..., nan_null: bool = ...): ...
