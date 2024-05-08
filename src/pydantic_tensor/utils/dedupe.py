from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def dedupe(values: Iterable[T]) -> Iterable[T]:
    seen = set[T]()
    for e in values:
        if e not in seen:
            yield e
            seen.add(e)
