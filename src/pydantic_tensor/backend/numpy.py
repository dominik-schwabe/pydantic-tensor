from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import TypeAlias, TypeGuard

from pydantic_tensor.utils.convert import ensure_int_tuple, ensure_str
from pydantic_tensor.utils.type import IoT, is_type

if TYPE_CHECKING:
    from types import ModuleType

    NumpyNDArray: TypeAlias = np.ndarray[Any, Any]
else:

    class NumpyNDArray:
        pass


class NumpyInterface:
    @staticmethod
    def get_name() -> str:
        return "numpy"

    @staticmethod
    def get_tensor_name() -> str:
        return "numpy.ndarray"

    @staticmethod
    def get_lib() -> ModuleType:
        return __import__("numpy")

    @staticmethod
    def is_imported() -> bool:
        return "numpy" in sys.modules

    @classmethod
    def get_tensor_type(cls):
        return cls.get_lib().ndarray

    @classmethod
    def is_tensor_type(cls, tp: Any) -> TypeGuard[IoT[np.ndarray[Any, Any]]]:
        return cls.is_imported() and is_type(tp, cls.get_lib().ndarray)

    @staticmethod
    def dtype_to_str(x: np.dtype[Any]) -> str:
        return ensure_str(x.name)

    @classmethod
    def str_to_dtype(cls, x: str) -> np.dtype[Any] | None:
        try:
            dtype = cls.get_lib().dtype(x)
        except TypeError:
            return None
        if dtype.isbuiltin != 1:
            return None
        return dtype

    @staticmethod
    def extract_shape(x: np.ndarray[Any, Any]) -> tuple[int, ...]:
        return ensure_int_tuple(x.shape)

    @staticmethod
    def extract_dtype(x: np.ndarray[Any, Any]) -> np.dtype[Any]:
        return x.dtype

    @staticmethod
    def get_dtype_alignment(x: np.dtype[Any]) -> tuple[int, int]:
        return (x.alignment, x.itemsize)

    @classmethod
    def from_numpy_view(cls, x: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
        return x.view(dtype)

    @staticmethod
    def to_numpy_view(x: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
        return x.view(dtype)
