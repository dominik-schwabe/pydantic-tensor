from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import TypeAlias, TypeGuard

from pydantic_tensor.utils.convert import ensure_int, ensure_int_tuple
from pydantic_tensor.utils.type import IoT, is_type

if TYPE_CHECKING:
    from types import ModuleType

    import torch

    TorchTensor: TypeAlias = torch.Tensor
else:

    class TorchTensor:
        pass


class TorchInterface:
    @staticmethod
    def get_tensor_name() -> str:
        return "torch.Tensor"

    @staticmethod
    def get_lib() -> ModuleType:
        return __import__("torch")

    @staticmethod
    def is_imported() -> bool:
        return "torch" in sys.modules

    @classmethod
    def get_tensor_type(cls) -> type[torch.Tensor]:
        return cls.get_lib().Tensor  # type: ignore[no-any-return]

    @classmethod
    def is_tensor_type(cls, tp: Any) -> TypeGuard[IoT[torch.Tensor]]:
        return cls.is_imported() and is_type(tp, cls.get_tensor_type())

    @staticmethod
    def dtype_to_str(x: torch.dtype) -> str:
        _, str_dtype = str(x).split(".")
        return str_dtype

    @classmethod
    def str_to_dtype(cls, x: str) -> torch.dtype | None:
        attr = getattr(cls.get_lib(), x, None)
        if not isinstance(attr, cls.get_lib().dtype):
            return None
        return attr

    @staticmethod
    def extract_shape(x: torch.Tensor) -> tuple[int, ...]:
        return ensure_int_tuple(x.shape)

    @staticmethod
    def extract_dtype(x: torch.Tensor) -> torch.dtype:
        return x.dtype

    @classmethod
    def get_dtype_alignment(cls, x: torch.dtype) -> tuple[int, int]:
        try:
            np_dtype = cls.get_lib().tensor([], dtype=x).numpy().dtype
        except TypeError:
            itemsize = ensure_int(x.itemsize)
            return (itemsize, itemsize)
        return (np_dtype.alignment, np_dtype.itemsize)

    @classmethod
    def from_numpy_view(cls, x: np.ndarray[Any, Any], dtype: torch.dtype) -> torch.Tensor:
        return cls.get_lib().tensor(x).view(dtype)

    @staticmethod
    def to_numpy_view(x: torch.Tensor, dtype: torch.dtype) -> np.ndarray[Any, Any]:
        return x.detach().view(dtype).numpy()
