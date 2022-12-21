from typing import Any

DISPATCHED_UFUNCS: set
REVERSED_NAMES: dict
UFUNC_ALIASES: dict
UNARY_UFUNCS: set

def maybe_dispatch_ufunc_to_dunder_op(*args, **kwargs) -> Any: ...
