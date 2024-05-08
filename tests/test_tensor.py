from __future__ import annotations

import warnings

from pydantic_tensor.types import Tensor_T

warnings.filterwarnings("ignore", ".*PyType_Spec with a metaclass that has custom.*")

import json
import sys
from contextlib import contextmanager
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from pydantic_tensor.backend.torch import TorchInterface, TorchTensor
from pydantic_tensor.delegate import NumpyDelegate
from pydantic_tensor.pydantic.dtype import ALL_DTYPES
from pydantic_tensor.tensor import ALL_INTERFACES, Tensor

LimitedInt = Annotated[int, Field(le=2)]

H = TypeVar("H", bound=int)
W = TypeVar("W", bound=int)


@contextmanager
def shadow_import(imports: list[str]):
    found: dict[str, Any] = {}
    try:
        for imp in imports:
            if imp in sys.modules:
                found[imp] = sys.modules[imp]
                del sys.modules[imp]
        yield
    finally:
        for k, v in found.items():
            if k not in sys.modules:
                sys.modules[k] = v


def convert(x: np.ndarray[Any, Any]):
    return json.dumps(NumpyDelegate.from_tensor(x, ALL_INTERFACES).serialize())


# spell-checker: disable
# convert(np.array(np.ones((2, 2), dtype="float32")))
FLOAT32_2X2 = '{"shape": [2, 2], "dtype": "float32", "data": "AACAPwAAgD8AAIA/AACAPw=="}'
# convert(np.array(np.ones((2, 3), dtype="float32")))
FLOAT32_2X3 = '{"shape": [2, 3], "dtype": "float32", "data": "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"}'


SERIALIZATIONS_2X2 = {dtype: json.loads(convert(np.array(np.ones((2, 2), dtype=dtype)))) for dtype in ALL_DTYPES}


def is_same(x: Any, y: Any):
    assert type(x) is type(y)
    assert np.array_equal(x, y)


def same_parse(
    ta: TypeAdapter[Tensor[Any, Any, Any]], vec: Any, ser: str, python_vec: Any = None, json_vec: Any = None
):
    parsed_vec = ta.validate_python(vec)
    python_dumped = ta.dump_python(parsed_vec)
    python_dumped_json_mode = ta.dump_python(parsed_vec, mode="json")
    json_dumped = ta.dump_json(parsed_vec)
    loaded = json.loads(json_dumped)
    python_vec_ = vec if python_vec is None else python_vec
    json_vec_ = vec if json_vec is None else json_vec
    is_same(python_dumped, python_vec_)
    assert loaded == ser
    is_same(ta.validate_python(python_dumped).value, python_vec_)
    is_same(ta.validate_python(python_dumped_json_mode).value, json_vec_)
    is_same(ta.validate_json(json_dumped).value, json_vec_)


def validate(tp: type[Tensor[Any, Any, Any]], json_str: str):
    ta = TypeAdapter(tp)
    ta.validate_json(json_str)


def validate_fail(tp: type[Tensor[Any, Any, Any]], json_str: str):
    ta = TypeAdapter(tp)
    with pytest.raises(ValidationError):
        ta.validate_json(json_str)


def test_dtype():
    for dtype, ser in SERIALIZATIONS_2X2.items():
        ta = TypeAdapter(Tensor[np.ndarray[Any, Any], Any, Literal[dtype]])
        np_ones = np.ones((2, 2), dtype=dtype)
        same_parse(ta, np_ones, ser)
        same_parse(ta, torch.ones((2, 2), dtype=TorchInterface.str_to_dtype(dtype)), ser, np_ones, np_ones)
        if dtype not in {"int64", "uint64", "float64", "complex64", "complex128"}:
            same_parse(ta, jnp.ones((2, 2), dtype=dtype), ser, np_ones, np_ones)
        if dtype not in {"int64", "uint64", "float64", "complex64", "complex128"}:
            same_parse(ta, tf.ones((2, 2), dtype=dtype), ser, np_ones, np_ones)


def test_shape():
    validate(Tensor[TorchTensor, tuple[int, int], Literal["float32"]], FLOAT32_2X2)
    validate(Tensor[TorchTensor, tuple[int, int], Literal["float32"]], FLOAT32_2X3)
    validate(Tensor[TorchTensor, tuple[int, LimitedInt], Literal["float32"]], FLOAT32_2X2)
    validate_fail(Tensor[TorchTensor, tuple[int, LimitedInt], Literal["float32"]], FLOAT32_2X3)
    validate_fail(Tensor[TorchTensor, tuple[int], Literal["float32"]], FLOAT32_2X2)


def test_missing_import():
    with shadow_import(["torch"]), pytest.raises(ValueError, match="no passed interface is loaded"):
        validate(Tensor[TorchTensor, tuple[int, int], Literal["float32"]], FLOAT32_2X2)
    validate(Tensor[Union[TorchTensor, np.ndarray[Any, Any]], tuple[int, int], Literal["float32"]], FLOAT32_2X2)
    with shadow_import(["torch"]):
        ta = TypeAdapter(Tensor[Union[TorchTensor, np.ndarray[Any, Any]], tuple[int, int], Literal["float32"]])
        np_ones = np.ones((2, 2), dtype="float32")
        same_parse(ta, tf.ones((2, 2), dtype=tf.float32), SERIALIZATIONS_2X2["float32"], np_ones, np_ones)


def test_generic():
    class PNGImage(BaseModel, Generic[Tensor_T, H, W]):
        content: Tensor[Tensor_T, tuple[H, W, Literal[3]], Any]
        type: Literal["png"] = "png"

    TypeAdapter(PNGImage[np.ndarray[Any, Any], int, int])
