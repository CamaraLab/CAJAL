import abc
from collections.abc import Callable as Callable
from pandas import Series as Series
from pandas._typing import Scalar as Scalar

class BaseStringArrayMethods(abc.ABC, metaclass=abc.ABCMeta): ...
