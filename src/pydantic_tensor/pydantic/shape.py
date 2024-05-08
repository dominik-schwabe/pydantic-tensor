from __future__ import annotations

from typing import Callable

from pydantic_core import core_schema


def apply_union(
    schema: core_schema.UnionSchema,
    func: Callable[[core_schema.CoreSchema], core_schema.CoreSchema],
):
    clean_choices: list[core_schema.CoreSchema | tuple[core_schema.CoreSchema, str]] = []
    for choice in schema["choices"]:
        if isinstance(choice, tuple):
            anno, label = choice
            clean_choices.append((func(anno), label))
        else:
            clean_choices.append(func(choice))
    return clean_choices


def postprocess_shape_element_schema(shape_anno: core_schema.CoreSchema):
    if shape_anno["type"] == "union":
        shape_anno["choices"] = apply_union(shape_anno, postprocess_shape_element_schema)
    elif shape_anno["type"] == "int":
        shape_anno["ge"] = max(0, shape_anno.get("ge", 0))
    elif shape_anno["type"] == "literal":
        for e in shape_anno["expected"]:
            if not isinstance(e, int):
                msg = f"shape literal {e} is not of type int"
                raise TypeError(msg)
            if e < 0:
                msg = f"shape literal {e} must not be negative"
                raise ValueError(msg)
    elif shape_anno["type"] == "any":
        shape_anno = {"type": "int", "ge": 0}
    else:
        msg = (
            f'type "{shape_anno["type"]}" is not supported for shape elements'
            ', only "Union", "int", "Literal" with int, and "Any" are allowed'
        )
        raise TypeError(msg)
    return shape_anno


def postprocess_shape_schema(shape_anno: core_schema.CoreSchema):
    if shape_anno["type"] == "union":
        shape_anno["choices"] = apply_union(shape_anno, postprocess_shape_schema)
    elif shape_anno["type"] == "tuple":
        shape_anno["items_schema"] = [postprocess_shape_element_schema(anno) for anno in shape_anno["items_schema"]]
    elif shape_anno["type"] == "any":
        shape_anno = {
            "type": "tuple",
            "items_schema": [{"type": "int", "ge": 0}],
            "variadic_item_index": 0,
        }
    else:
        msg = (
            f'type "{shape_anno["type"]}" is not supported for shape tuples'
            ', only "Union", "tuple", and "Any" are allowed'
        )
        raise TypeError(msg)
    return shape_anno
