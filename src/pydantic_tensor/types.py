from __future__ import annotations

from typing import Literal, TypedDict, TypeVar, Union

from pydantic_tensor.backend.jax import JaxArray
from pydantic_tensor.backend.numpy import NumpyNDArray
from pydantic_tensor.backend.tensorflow import TensorflowTensor
from pydantic_tensor.backend.torch import TorchTensor


class JSONTensor(TypedDict):
    shape: tuple[int, ...]
    dtype: str
    data: str


TensorTypes = Union[NumpyNDArray, TorchTensor, TensorflowTensor, JaxArray]

Int = Literal["int8", "int16", "int32", "int64"]
UInt = Literal["uint8", "uint16", "uint32", "uint64"]
Float = Literal["float16", "float32", "float64"]
Complex = Literal["complex64", "complex128"]
BFloat = Literal["bfloat16"]

DTypes = Union[Int, UInt, Float, Complex, BFloat]

Tensor_T = TypeVar("Tensor_T", bound=TensorTypes)
Shape_T = TypeVar("Shape_T", bound=tuple[int, ...])
DType_T = TypeVar("DType_T", bound=DTypes)
