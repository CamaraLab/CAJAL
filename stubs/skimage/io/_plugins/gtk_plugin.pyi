import gtk
from .util import GuiLockError as GuiLockError, prepare_for_display as prepare_for_display, window_manager as window_manager
from _typeshed import Incomplete

class ImageWindow(gtk.Window):
    mgr: Incomplete
    img: Incomplete
    def __init__(self, arr, mgr) -> None: ...
    def destroy(self, widget, data: Incomplete | None = ...) -> None: ...

def imshow(arr) -> None: ...
