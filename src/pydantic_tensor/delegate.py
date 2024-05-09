from __future__ import annotations

import base64
import sys
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from operator import __mul__
from typing import Any

import numpy as np

from pydantic_tensor.interface import TensorInterface
from pydantic_tensor.types import JSONTensor
from pydantic_tensor.utils.base64 import base64_num_bytes
from pydantic_tensor.utils.convert import ensure_int_tuple

BYTEORDER = {
    "=": sys.byteorder,
    "<": "little",
    ">": "big",
    "|": None,
}


def convert_endianness(vec: np.ndarray[Any, Any]):
    byteorder = BYTEORDER[vec.dtype.byteorder]
    if byteorder == "big":
        vec = vec.byteswap()
    return vec


def score_candidate(candidate: str):
    if candidate.startswith("complex"):
        return 3
    if candidate.startswith("float"):
        return 2
    if candidate.startswith("int"):
        return 1
    return 0


def select_best(candidates: Iterable[np.dtype[Any]]):
    _, best = max([(score_candidate(e.name), e) for e in candidates], key=lambda x: x[0])
    return best


DELEGATE_CANDIDATES = defaultdict[tuple[int, int], set[np.dtype[Any]]](set)
for v in np.sctypeDict.values():
    dtp = np.dtype(v)
    if dtp.isbuiltin == 1 and dtp.alignment <= dtp.itemsize and (dtp.byteorder != "|" or dtp.itemsize == 1):
        DELEGATE_CANDIDATES[(dtp.alignment, dtp.itemsize)].add(dtp)


ALIGNMENT_TO_DTYPE_DELEGATE = {align: select_best(candidates) for align, candidates in DELEGATE_CANDIDATES.items()}


def iter_loaded_interface_by_dtype(interfaces: Iterable[TensorInterface[Any, Any]], dtype: str):
    loaded: list[TensorInterface[Any, Any]] = []
    yielded = False
    for interface in interfaces:
        if interface.is_imported():
            resolved = interface.str_to_dtype(dtype)
            if resolved is not None:
                yielded = True
                yield interface, resolved
            loaded.append(interface)
    if not yielded:
        if not loaded:
            msg = f"no passed interface is loaded: {[e.get_tensor_name() for e in interfaces]}"
        else:
            msg = f'no passed loaded interface supports dtype "{dtype}"'
        raise ValueError(msg)


def find_interface_by_tensor(tensor: Any, interfaces: Iterable[TensorInterface[Any, Any]]):
    for interface in interfaces:
        if interface.is_tensor_type(tensor):
            return interface
    supported = [tif.get_tensor_name() for tif in interfaces]
    tp = type(tensor)
    msg = f'no passed interface handles tensor "{tp.__module__}.{tp.__name__}" (interfaces support: {supported})'
    raise ValueError(msg)


class NumpyDelegate:
    def __init__(
        self, delegate: np.ndarray[Any, Any], original_dtype: str, source_interface: TensorInterface[Any, Any]
    ):
        self.delegate = delegate
        self.original_dtype = original_dtype
        self.source_interface = source_interface

    @classmethod
    def from_json_tensor(cls, x: JSONTensor, interfaces: list[TensorInterface[Any, Any]]) -> NumpyDelegate:
        data, shape, dtype = x["data"], x["shape"], x["dtype"]
        interface, resolved_dtype = next(iter_loaded_interface_by_dtype(interfaces, dtype))
        align, itemsize = interface.get_dtype_alignment(resolved_dtype)
        dtype_delegate = ALIGNMENT_TO_DTYPE_DELEGATE[(align, itemsize)]
        num_bytes = base64_num_bytes(data)
        num_data_elements = num_bytes / itemsize
        num_shape_elements = reduce(__mul__, shape)
        if num_shape_elements != num_data_elements:
            debug_info = {
                "itemsize": itemsize,
                "num_shape_elements": num_shape_elements,
                "num_data_elements": num_data_elements,
            }
            msg = f"the number of shape and data elements are different {debug_info!s}"
            raise ValueError(msg)
        vec = convert_endianness(np.frombuffer(base64.b64decode(data), dtype=dtype_delegate).reshape(shape))
        return cls(vec, dtype, interface)

    @classmethod
    def from_tensor(cls, x: Any, interfaces: list[TensorInterface[Any, Any]]):
        interface = find_interface_by_tensor(x, interfaces)
        shape = interface.extract_shape(x)
        dtype = interface.extract_dtype(x)
        alignment = interface.get_dtype_alignment(dtype)
        dtype_delegate = ALIGNMENT_TO_DTYPE_DELEGATE[alignment]
        native_dtype_delegate = interface.str_to_dtype(dtype_delegate.name)
        if native_dtype_delegate is None:
            msg = f'dtype {dtype_delegate} is not supported by "{interface.get_tensor_name()}"'
            raise ValueError(msg)
        numpy_tensor = interface.to_numpy_view(x, native_dtype_delegate)
        serialized_dtype = numpy_tensor.dtype.name
        requested_dtype = dtype_delegate.name
        if requested_dtype != serialized_dtype:
            msg = f'requested dtype "{requested_dtype}" but serialized as "{serialized_dtype}"'
            raise ValueError(msg)
        serialized_shape = ensure_int_tuple(numpy_tensor.shape)
        if shape != serialized_shape:
            msg = f"requested shape {shape} but serialized as {serialized_shape}"
            raise ValueError(msg)
        return cls(convert_endianness(numpy_tensor), interface.dtype_to_str(dtype), interface)

    def serialize(self) -> JSONTensor:
        return {
            "shape": tuple(self.delegate.shape),
            "dtype": self.original_dtype,
            "data": base64.b64encode(bytes(self.delegate)).decode(),
        }

    def deserialize(self, target_interface: TensorInterface[Any, Any] | None = None):
        tif = self.source_interface if target_interface is None else target_interface
        dtype = tif.str_to_dtype(self.original_dtype)
        if dtype is None:
            msg = f'interface for "{tif.get_tensor_name()}" can not handle dtype "{self.original_dtype}"'
            raise ValueError(msg)
        deserialized = tif.from_numpy_view(self.delegate, dtype)
        serialized_dtype = tif.dtype_to_str(tif.extract_dtype(deserialized))
        requested_dtype = self.original_dtype
        if self.original_dtype != serialized_dtype:
            msg = f'requested dtype "{requested_dtype}" but deserialized as "{serialized_dtype}"'
            raise ValueError(msg)
        serialized_shape = tif.extract_shape(deserialized)
        requested_shape = self.delegate.shape
        if requested_shape != serialized_shape:
            msg = f"requested shape {requested_shape} but deserialized as {serialized_shape}"
            raise ValueError(msg)
        return deserialized
