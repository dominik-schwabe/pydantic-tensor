from __future__ import annotations

import contextlib
import types
import typing
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union

UNION_TYPES = {Union}

if not TYPE_CHECKING:
    with contextlib.suppress(AttributeError):
        UNION_TYPES.add(types.UnionType)

T = TypeVar("T")


class NoTypeAnnotationFoundError(Exception):
    pass


def is_annotated(tp: type[Any]) -> bool:
    return typing.get_origin(tp) is not None


def find_generics(origin: type[Any]) -> tuple[Any, ...]:
    for base in getattr(origin, "__orig_bases__", ()):
        if typing.get_origin(base) is Generic:
            return typing.get_args(base)
    return ()


def extract_annotated(tp: type[Any], query_tp: type[Any]) -> list[Any] | None:
    origin = typing.get_origin(tp)
    if origin:
        # support types that don't inherit from typing.Generic (e.g. tuple, list, set, dict)
        if not hasattr(origin, "__orig_bases__") and origin is query_tp:
            return list(typing.get_args(tp))
        else:
            anno = extract_non_annotated(origin, query_tp)
            if anno is not None:
                rep_map = dict(zip(find_generics(origin), typing.get_args(tp)))
                filtered: list[Any] = []
                for e in anno:
                    rep = rep_map.get(e)
                    if rep is None and not isinstance(e, TypeVar):
                        rep = e
                    filtered.append(rep)
                return filtered
    return None


def extract_non_annotated(tp: type[Any], query_tp: type[Any]) -> list[Any] | None:
    if tp is query_tp:
        return list(find_generics(tp))
    else:
        for base in getattr(tp, "__orig_bases__", ()):
            anno = extract_annotated(base, query_tp) if is_annotated(base) else extract_non_annotated(base, query_tp)
            if anno is not None:
                return anno
    return None


def extract_type_annotation(tp: type[Any], query_tp: type[Any]) -> list[type[Any] | None]:
    result = extract_annotated(tp, query_tp) if is_annotated(tp) else extract_non_annotated(tp, query_tp)
    if result is None:
        raise NoTypeAnnotationFoundError
    return [None if isinstance(e, TypeVar) else e for e in result]


def default_any(v: T | None) -> type[Any] | T:
    return object if v is None else v


def extract_literals(anno: Any) -> list[Any]:
    origin = typing.get_origin(anno)
    if origin in UNION_TYPES:
        return [e for args in typing.get_args(anno) for e in extract_literals(args)]
    if origin is Literal:
        return list(typing.get_args(anno))
    msg = f'can not extract literals from "{origin}"'
    raise ValueError(msg)


def extract_union(anno: type[Any]):
    if typing.get_origin(anno) in UNION_TYPES:
        return typing.get_args(anno)
    return [anno]
