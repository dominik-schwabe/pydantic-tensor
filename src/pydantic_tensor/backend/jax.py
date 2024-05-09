from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import TypeAlias, TypeGuard

from pydantic_tensor.utils.convert import ensure_int_tuple
from pydantic_tensor.utils.type import IoT, is_type

if TYPE_CHECKING:
    from types import ModuleType

    import jax
    import jax.numpy as jnp

    JaxArray: TypeAlias = jax.Array
else:

    class JaxArray:
        pass


class JaxInterface:
    @staticmethod
    def get_name() -> str:
        return "jax"

    @staticmethod
    def get_tensor_name() -> str:
        return "jax.Array"

    @staticmethod
    def is_imported() -> bool:
        return "jax" in sys.modules or "jax.numpy" in sys.modules

    @staticmethod
    def get_lib() -> ModuleType:
        return __import__("jax").numpy  # type: ignore[no-any-return]

    @classmethod
    def get_tensor_type(cls) -> type[jax.Array]:
        return cls.get_lib().ndarray  # type: ignore[no-any-return]

    @classmethod
    def is_tensor_type(cls, tp: Any) -> TypeGuard[IoT[jax.Array]]:
        return cls.is_imported() and is_type(tp, cls.get_tensor_type())

    @staticmethod
    def dtype_to_str(x: jnp.dtype[Any]) -> str:
        return str(x)

    @classmethod
    def str_to_dtype(cls, x: str) -> jnp.dtype[Any] | None:
        try:
            return cls.get_lib().dtype(x)
        except TypeError:
            return None

    @staticmethod
    def extract_shape(x: jax.Array) -> tuple[int, ...]:
        return ensure_int_tuple(x.shape)

    @staticmethod
    def extract_dtype(x: jax.Array) -> jnp.dtype[Any]:
        return x.dtype

    @staticmethod
    def get_dtype_alignment(x: jnp.dtype[Any]) -> tuple[int, int]:
        return (x.alignment, x.itemsize)

    @classmethod
    def from_numpy_view(cls, x: np.ndarray[Any, Any], dtype: jnp.dtype[Any]) -> jax.Array:
        return cls.get_lib().asarray(x).view(dtype)

    @staticmethod
    def to_numpy_view(x: jax.Array, dtype: jnp.dtype[Any]) -> np.ndarray[Any, Any]:
        return np.asarray(x.view(dtype))
