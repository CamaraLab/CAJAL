from .q_color_mixer import MixerPanel as MixerPanel
from .q_histogram import QuadHistogram as QuadHistogram
from _typeshed import Incomplete
from qtpy.QtWidgets import QFrame, QLabel, QMainWindow

class ImageLabel(QLabel):
    parent: Incomplete
    arr: Incomplete
    img: Incomplete
    pm: Incomplete
    def __init__(self, parent, arr) -> None: ...
    def mouseMoveEvent(self, evt) -> None: ...
    def resizeEvent(self, evt) -> None: ...
    def update_image(self) -> None: ...

class RGBHSVDisplay(QFrame):
    posx_label: Incomplete
    posx_value: Incomplete
    posy_label: Incomplete
    posy_value: Incomplete
    r_label: Incomplete
    r_value: Incomplete
    g_label: Incomplete
    g_value: Incomplete
    b_label: Incomplete
    b_value: Incomplete
    h_label: Incomplete
    h_value: Incomplete
    s_label: Incomplete
    s_value: Incomplete
    v_label: Incomplete
    v_value: Incomplete
    layout: Incomplete
    def __init__(self) -> None: ...
    def update_vals(self, data) -> None: ...

class SkiviImageWindow(QMainWindow):
    arr: Incomplete
    mgr: Incomplete
    main_widget: Incomplete
    layout: Incomplete
    label: Incomplete
    label_container: Incomplete
    mixer_panel: Incomplete
    rgbv_hist: Incomplete
    rgb_hsv_disp: Incomplete
    save_file: Incomplete
    save_stack: Incomplete
    def __init__(self, arr, mgr) -> None: ...
    def closeEvent(self, event) -> None: ...
    def update_histograms(self) -> None: ...
    def refresh_image(self) -> None: ...
    def scale_mouse_pos(self, x, y): ...
    def label_mouseMoveEvent(self, evt) -> None: ...
    def save_to_stack(self) -> None: ...
    def save_to_file(self) -> None: ...
