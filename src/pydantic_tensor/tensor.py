from __future__ import annotations

from typing import Any, Generic

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, SerializationInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import SchemaValidator, core_schema

from pydantic_tensor.delegate import NumpyDelegate, find_interface_by_tensor, find_loaded_interface_by_dtype
from pydantic_tensor.interface import TENSOR_TYPE_TO_INTERFACE, annotation_to_interfaces
from pydantic_tensor.pydantic.dtype import build_dtype_schema
from pydantic_tensor.pydantic.shape import postprocess_shape_schema
from pydantic_tensor.types import DType_T, JSONTensor, Shape_T, Tensor_T
from pydantic_tensor.utils.dedupe import dedupe
from pydantic_tensor.utils.type_annotation import default_any, extract_type_annotation

ALL_INTERFACES = list(TENSOR_TYPE_TO_INTERFACE.values())


class Tensor(Generic[Tensor_T, Shape_T, DType_T]):
    def __init__(self, value: Tensor_T):
        self.value = value

    @staticmethod
    def __get_pydantic_core_schema__(source: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        tensor_anno, shape_anno, dtype_anno = map(default_any, extract_type_annotation(source, Tensor))
        interfaces = list(dedupe(annotation_to_interfaces(tensor_anno)))
        shape_schema = postprocess_shape_schema(handler(shape_anno))
        dtype_schema = build_dtype_schema(dtype_anno)

        shape_validator = SchemaValidator(shape_schema)
        dtype_validator = SchemaValidator(dtype_schema)

        def deserialize(x: Any) -> Tensor[Tensor_T, Shape_T, DType_T]:
            return Tensor(NumpyDelegate.from_json_tensor(x, interfaces).deserialize())

        def serialize_tensor(x: Tensor[Tensor_T, Shape_T, DType_T], info: SerializationInfo) -> JSONTensor:
            if "json" in info.mode:
                json_tensor: JSONTensor = NumpyDelegate.from_tensor(x.value, interfaces).serialize()
                return json_tensor
            return x.value

        def validate_tensor(x: Any) -> Tensor[Tensor_T, Shape_T, DType_T]:
            if isinstance(x, Tensor):
                x = x.value
            tif = find_interface_by_tensor(x, ALL_INTERFACES)
            str_dtype = tif.dtype_to_str(tif.extract_dtype(x))
            shape_validator.validate_python(tif.extract_shape(x))
            dtype_validator.validate_python(str_dtype)
            if tif not in interfaces:
                handler_tif, _ = find_loaded_interface_by_dtype(interfaces, str_dtype)
                x = NumpyDelegate.from_tensor(x, [tif]).deserialize(handler_tif)
            return Tensor(x)

        json_schema = core_schema.no_info_after_validator_function(
            deserialize,
            core_schema.typed_dict_schema(
                {
                    "shape": core_schema.typed_dict_field(shape_schema),
                    "dtype": core_schema.typed_dict_field(dtype_schema),
                    "data": core_schema.typed_dict_field(core_schema.str_schema()),
                }
            ),
        )

        python_schema = core_schema.union_schema(
            [core_schema.no_info_plain_validator_function(validate_tensor), json_schema],
        )

        return core_schema.json_or_python_schema(
            python_schema=python_schema,
            json_schema=json_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize_tensor, info_arg=True),
        )

    @staticmethod
    def __get_pydantic_json_schema__(
        core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema["properties"]["dtype"]["format"] = "base64"
        return json_schema

    def __repr__(self) -> str:
        return f"Tensor({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)
