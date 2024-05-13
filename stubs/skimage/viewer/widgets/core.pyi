from ..qt import QtWidgets
from _typeshed import Incomplete

class BaseWidget(QtWidgets.QWidget):
    plugin: Incomplete
    name: Incomplete
    ptype: Incomplete
    callback: Incomplete
    def __init__(self, name, ptype: Incomplete | None = ..., callback: Incomplete | None = ...) -> None: ...
    @property
    def val(self) -> None: ...

class Text(BaseWidget):
    layout: Incomplete
    def __init__(self, name: Incomplete | None = ..., text: str = ...) -> None: ...
    @property
    def text(self): ...
    @text.setter
    def text(self, text_str) -> None: ...

class Slider(BaseWidget):
    slider: Incomplete
    layout: Incomplete
    value_fmt: str
    value_type: Incomplete
    name_label: Incomplete
    editbox: Incomplete
    def __init__(self, name, low: float = ..., high: float = ..., value: Incomplete | None = ..., value_type: str = ..., ptype: str = ..., callback: Incomplete | None = ..., max_edit_width: int = ..., orientation: str = ..., update_on: str = ...) -> None: ...
    @property
    def val(self): ...
    @val.setter
    def val(self, value) -> None: ...

class ComboBox(BaseWidget):
    name_label: Incomplete
    layout: Incomplete
    def __init__(self, name, items, ptype: str = ..., callback: Incomplete | None = ...) -> None: ...
    @property
    def val(self): ...
    @property
    def index(self): ...
    @index.setter
    def index(self, i) -> None: ...

class CheckBox(BaseWidget):
    layout: Incomplete
    def __init__(self, name, value: bool = ..., alignment: str = ..., ptype: str = ..., callback: Incomplete | None = ...) -> None: ...
    @property
    def val(self): ...
    @val.setter
    def val(self, i) -> None: ...

class Button(BaseWidget):
    layout: Incomplete
    def __init__(self, name, callback) -> None: ...
