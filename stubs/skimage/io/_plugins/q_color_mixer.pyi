from .util import ColorMixer as ColorMixer
from _typeshed import Incomplete
from qtpy.QtWidgets import QFrame, QWidget

class IntelligentSlider(QWidget):
    name: Incomplete
    callback: Incomplete
    a: Incomplete
    b: Incomplete
    manually_triggered: bool
    slider: Incomplete
    name_label: Incomplete
    value_label: Incomplete
    layout: Incomplete
    def __init__(self, name, a, b, callback) -> None: ...
    def slider_changed(self, val) -> None: ...
    def set_conv_fac(self, a, b) -> None: ...
    def set_value(self, val) -> None: ...
    def val(self): ...

class MixerPanel(QFrame):
    img: Incomplete
    mixer: Incomplete
    callback: Incomplete
    combo_box_entries: Incomplete
    combo_box: Incomplete
    rgb_add: Incomplete
    rgb_mul: Incomplete
    rs: Incomplete
    gs: Incomplete
    bs: Incomplete
    rgb_widget: Incomplete
    hsv_add: Incomplete
    hsv_mul: Incomplete
    hs: Incomplete
    ss: Incomplete
    vs: Incomplete
    hsv_widget: Incomplete
    cont: Incomplete
    bright: Incomplete
    bright_widget: Incomplete
    gamma: Incomplete
    gamma_widget: Incomplete
    a_gamma: Incomplete
    b_gamma: Incomplete
    sig_gamma_widget: Incomplete
    commit_button: Incomplete
    revert_button: Incomplete
    sliders: Incomplete
    layout: Incomplete
    def __init__(self, img) -> None: ...
    def set_callback(self, callback) -> None: ...
    def combo_box_changed(self, index) -> None: ...
    def rgb_radio_changed(self) -> None: ...
    def hsv_radio_changed(self) -> None: ...
    def reset(self) -> None: ...
    def reset_sliders(self) -> None: ...
    def rgb_changed(self, name, val) -> None: ...
    def hsv_changed(self, name, val) -> None: ...
    def bright_changed(self, name, val) -> None: ...
    def gamma_changed(self, name, val) -> None: ...
    def sig_gamma_changed(self, name, val) -> None: ...
    def commit_changes(self) -> None: ...
    def revert_changes(self) -> None: ...
