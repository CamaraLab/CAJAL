from .util import histograms as histograms
from _typeshed import Incomplete
from qtpy.QtWidgets import QFrame, QWidget

class ColorHistogram(QWidget):
    counts: Incomplete
    n: Incomplete
    colormap: Incomplete
    def __init__(self, counts, colormap) -> None: ...
    def paintEvent(self, evt) -> None: ...
    def update_hist(self, counts, cmap) -> None: ...

class QuadHistogram(QFrame):
    r_hist: Incomplete
    g_hist: Incomplete
    b_hist: Incomplete
    v_hist: Incomplete
    layout: Incomplete
    def __init__(self, img, layout: str = ..., order=...) -> None: ...
    def update_hists(self, img) -> None: ...
