from __future__ import annotations

import typing
from types import ModuleType
from typing import Any, Protocol, TypeVar

import numpy as np
from typing_extensions import TypeGuard

from pydantic_tensor.backend.jax import JaxArray, JaxInterface
from pydantic_tensor.backend.numpy import NumpyInterface, NumpyNDArray
from pydantic_tensor.backend.tensorflow import TensorflowInterface, TensorflowTensor
from pydantic_tensor.backend.torch import TorchInterface, TorchTensor
from pydantic_tensor.utils.type import IoT
from pydantic_tensor.utils.type_annotation import extract_union

TENSOR_T = TypeVar("TENSOR_T")
DTYPE_T = TypeVar("DTYPE_T")


class TensorInterface(Protocol[TENSOR_T, DTYPE_T]):
    @staticmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    def get_tensor_name() -> str:
        raise NotImplementedError

    @staticmethod
    def is_imported() -> bool:
        raise NotImplementedError

    @staticmethod
    def get_lib() -> ModuleType:
        raise NotImplementedError

    @staticmethod
    def is_tensor_type(tp: Any) -> TypeGuard[IoT[TENSOR_T]]:
        raise NotImplementedError

    @staticmethod
    def dtype_to_str(x: DTYPE_T) -> str:
        raise NotImplementedError

    @staticmethod
    def str_to_dtype(x: str) -> DTYPE_T | None:
        raise NotImplementedError

    @staticmethod
    def extract_shape(x: TENSOR_T) -> tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def extract_dtype(x: TENSOR_T) -> DTYPE_T:
        raise NotImplementedError

    @staticmethod
    def get_dtype_alignment(x: DTYPE_T) -> tuple[int, int]:
        raise NotImplementedError

    @staticmethod
    def from_numpy_view(x: np.ndarray[Any, Any], dtype: DTYPE_T) -> TENSOR_T:
        raise NotImplementedError

    @staticmethod
    def to_numpy_view(x: TENSOR_T, dtype: DTYPE_T) -> np.ndarray[Any, Any]:
        raise NotImplementedError


TENSOR_TYPE_TO_INTERFACE: dict[Any, type[TensorInterface[Any, Any]]] = {
    NumpyNDArray: NumpyInterface,
    TorchTensor: TorchInterface,
    TensorflowTensor: TensorflowInterface,
    JaxArray: JaxInterface,
}

SUPPORTED_TENSOR_TYPES = [
    *(e.get_tensor_name() for e in TENSOR_TYPE_TO_INTERFACE.values()),
    *(e.__name__ for e in TENSOR_TYPE_TO_INTERFACE),
]


def type_to_interface(tensor_type: type[Any]) -> type[TensorInterface[Any, Any]]:
    if tensor_type in TENSOR_TYPE_TO_INTERFACE:
        return TENSOR_TYPE_TO_INTERFACE[tensor_type]
    for interface in TENSOR_TYPE_TO_INTERFACE.values():
        if interface.is_tensor_type(tensor_type):
            return interface
    msg = f"{tensor_type} is not a supported tensor type, use one of {SUPPORTED_TENSOR_TYPES}"
    raise TypeError(msg)


ANY_TYPES = [object, typing.Any]


def annotation_to_interfaces(tensor_anno: type[Any]):
    tensor_types = [typing.get_origin(e) or e for e in extract_union(tensor_anno)]
    if any(e in ANY_TYPES for e in tensor_types):
        tensor_types = [e for e in tensor_types if e not in ANY_TYPES]
        tensor_types.extend(TENSOR_TYPE_TO_INTERFACE.keys())
    return [type_to_interface(e) for e in tensor_types]
