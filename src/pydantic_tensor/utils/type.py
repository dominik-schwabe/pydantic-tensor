from __future__ import annotations

from typing import Any, TypeVar, Union

from typing_extensions import TypeGuard

T = TypeVar("T")

IoT = Union[T, type[T]]  # Instance or Type


def is_type(tp: Any, tensor_type: type[T]) -> TypeGuard[IoT[T]]:
    return issubclass(tp, tensor_type) if isinstance(tp, type) else isinstance(tp, tensor_type)
