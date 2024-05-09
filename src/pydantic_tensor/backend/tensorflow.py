from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import TypeAlias, TypeGuard

from pydantic_tensor.utils.convert import ensure_int_tuple, ensure_str
from pydantic_tensor.utils.type import IoT, is_type

if TYPE_CHECKING:
    from types import ModuleType

    import tensorflow as tf

    TensorflowTensor: TypeAlias = tf.Tensor
else:

    class TensorflowTensor:
        pass


class TensorflowInterface:
    @staticmethod
    def get_name() -> str:
        return "tensorflow"

    @staticmethod
    def get_tensor_name() -> str:
        return "tensorflow.Tensor"

    @staticmethod
    def is_imported() -> bool:
        return "tensorflow" in sys.modules

    @staticmethod
    def get_lib() -> ModuleType:
        return __import__("tensorflow")

    @classmethod
    def get_tensor_type(cls) -> type[tf.Tensor]:
        return cls.get_lib().Tensor  # type: ignore[no-any-return]

    @classmethod
    def is_tensor_type(cls, tp: Any) -> TypeGuard[IoT[tf.Tensor]]:
        return cls.is_imported() and is_type(tp, cls.get_tensor_type())

    @staticmethod
    def dtype_to_str(x: tf.DType) -> str:
        return ensure_str(x.name)

    @classmethod
    def str_to_dtype(cls, x: str) -> tf.DType | None:
        attr = getattr(cls.get_lib(), x, None)
        if not isinstance(attr, cls.get_lib().DType):
            return None
        return attr

    @staticmethod
    def extract_shape(x: tf.Tensor) -> tuple[int, ...]:
        return ensure_int_tuple(x.shape)

    @staticmethod
    def extract_dtype(x: tf.Tensor) -> tf.DType:
        return x.dtype

    @staticmethod
    def get_dtype_alignment(x: tf.DType) -> tuple[int, int]:
        dtp = np.dtype(x.as_numpy_dtype)
        return (dtp.alignment, dtp.itemsize)

    @classmethod
    def from_numpy_view(cls, x: np.ndarray[Any, Any], dtype: tf.DType) -> tf.Tensor:
        return cls.get_lib().convert_to_tensor(x.view(dtype.as_numpy_dtype))

    @classmethod
    def to_numpy_view(cls, x: tf.Tensor, dtype: tf.DType) -> np.ndarray[Any, Any]:
        return x.numpy().view(cls.dtype_to_str(dtype))
