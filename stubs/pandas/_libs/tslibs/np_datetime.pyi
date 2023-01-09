from typing import Any

class OutOfBoundsDatetime(ValueError): ...

class OutOfBoundsTimedelta(ValueError): ...

def astype_overflowsafe(*args, **kwargs) -> Any: ...
def compare_mismatched_resolutions(*args, **kwargs) -> Any: ...
def is_unitless(*args, **kwargs) -> Any: ...
def py_get_unit_from_dtype(*args, **kwargs) -> Any: ...
def py_td64_to_tdstruct(*args, **kwargs) -> Any: ...