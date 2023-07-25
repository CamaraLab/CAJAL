from . import misc as misc
from ... import util as util
from ..path import Path as Path
from .svg_io import svg_to_path as svg_to_path
from _typeshed import Incomplete

def load_path(file_obj, file_type: Incomplete | None = ..., **kwargs): ...
def path_formats(): ...

path_loaders: Incomplete
