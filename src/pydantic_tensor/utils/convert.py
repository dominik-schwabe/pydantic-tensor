from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def ensure_int_tuple(values: Iterable[Any]) -> tuple[int, ...]:
    int_shape: list[int] = []
    for e in values:
        if not isinstance(e, int):
            msg = f"element {e} is not of type int"
            raise TypeError(msg)
        int_shape.append(e)
    return tuple(int_shape)


def ensure_int(value: Any) -> int:
    if not isinstance(value, int):
        msg = f"value {value} is not of type int"
        raise TypeError(msg)
    return value


def ensure_str(value: Any) -> str:
    if not isinstance(value, str):
        msg = f"value {value} is not of type str"
        raise TypeError(msg)
    return value
