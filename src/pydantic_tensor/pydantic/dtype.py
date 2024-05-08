from __future__ import annotations

import typing
from typing import Any, ForwardRef, Literal

from pydantic_core import core_schema

from pydantic_tensor.types import DTypes
from pydantic_tensor.utils.dedupe import dedupe
from pydantic_tensor.utils.type_annotation import extract_literals, extract_union

ALL_DTYPES = extract_literals(DTypes)
ALL_DTYPES_SET = set(ALL_DTYPES)


def build_dtype_schema(dtype_anno: type[Any]):
    flat_dtypes: list[Any] = []
    for e in extract_union(dtype_anno):
        if typing.get_origin(e) is Literal:
            flat_dtypes.extend(typing.get_args(e))
        else:
            flat_dtypes.append(e)
    resolved_dtypes: list[str] = []
    for dtype in flat_dtypes:
        if dtype is Any:
            return core_schema.literal_schema(ALL_DTYPES)
        elif dtype in ALL_DTYPES_SET:
            resolved_dtypes.append(dtype)
        elif isinstance(dtype, ForwardRef):
            msg = f"invalid {dtype} (hint: wrap it in a typing.Literal)"
            raise TypeError(msg)
        else:
            msg = f"unknown dtype {dtype}"
            raise TypeError(msg)

    return core_schema.literal_schema(list(dedupe(resolved_dtypes)))
