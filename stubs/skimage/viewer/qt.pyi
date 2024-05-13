from _typeshed import Incomplete
from matplotlib.backends.qt4_compat import QtGui

has_qt: bool
QtWidgets = QtGui

class QtGui_cls:
    QMainWindow: Incomplete
    QDialog: Incomplete
    QWidget: Incomplete

class QtCore_cls:
    class Qt:
        TopDockWidgetArea: Incomplete
        BottomDockWidgetArea: Incomplete
        LeftDockWidgetArea: Incomplete
        RightDockWidgetArea: Incomplete
    def Signal(self, *args, **kwargs) -> None: ...
FigureManagerQT = object
FigureCanvasQTAgg = object
Qt: Incomplete
Signal: Incomplete
