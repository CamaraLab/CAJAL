from _typeshed import Incomplete
from typing import Hashable, Literal

class OutputKey:
    label: Hashable
    position: int
    def __init__(self, label, position) -> None: ...
    def __lt__(self, other): ...
    def __gt__(self, other): ...
    def __le__(self, other): ...
    def __ge__(self, other): ...

plotting_methods: Incomplete
common_apply_allowlist: Incomplete
series_apply_allowlist: frozenset[str]
dataframe_apply_allowlist: frozenset[str]
cythonized_kernels: Incomplete
reduction_kernels: Incomplete

def maybe_normalize_deprecated_kernels(kernel) -> Literal['bfill', 'ffill']: ...

transformation_kernels: Incomplete
groupby_other_methods: Incomplete
transform_kernel_allowlist: Incomplete
