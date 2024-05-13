from .._shared import utils as utils
from .._shared.utils import warn as warn
from ..util import img_as_float as img_as_float
from _typeshed import Incomplete

old_del: Incomplete

def new_del(self) -> None: ...

UmfpackContext: Incomplete
amg_loaded: bool

def random_walker(data, labels, beta: int = ..., mode: str = ..., tol: float = ..., copy: bool = ..., multichannel: bool = ..., return_full_prob: bool = ..., spacing: Incomplete | None = ..., *, prob_tol: float = ..., channel_axis: Incomplete | None = ...): ...
