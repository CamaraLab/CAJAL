from ..._shared.utils import warn as warn
from .util import prepare_for_display as prepare_for_display, window_manager as window_manager
from _typeshed import Incomplete
from qtpy.QtWidgets import QLabel, QMainWindow

app: Incomplete

class ImageLabel(QLabel):
    arr: Incomplete
    img: Incomplete
    pm: Incomplete
    def __init__(self, parent, arr) -> None: ...
    def resizeEvent(self, evt) -> None: ...

class ImageWindow(QMainWindow):
    mgr: Incomplete
    main_widget: Incomplete
    layout: Incomplete
    label: Incomplete
    def __init__(self, arr, mgr) -> None: ...
    def closeEvent(self, event) -> None: ...

def imread(filename): ...
def imshow(arr, fancy: bool = ...) -> None: ...
def imsave(filename, img, format_str: Incomplete | None = ...) -> None: ...
