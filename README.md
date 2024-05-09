# pydantic-tensor

Support parsing, validation, and serialization of common tensors (`np.ndarray`, `torch.Tensor`, `tensorflow.Tensor`, `jax.Array`) for Pydantic.

[![PyPI - Version](https://img.shields.io/pypi/v/pydantic-tensor.svg)](https://pypi.org/project/pydantic-tensor)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pydantic-tensor.svg)](https://pypi.org/project/pydantic-tensor)

---

## Installation

```console
pip install pydantic-tensor
```

## Usage

### Validation

```python
from typing import Annotated, Any, Literal

import numpy as np
import tensorflow as tf
import torch
from pydantic import BaseModel, Field

from pydantic_tensor import Tensor

# allow only integers greater equal than 2 and less equal than 3
DimType = Annotated[int, Field(ge=2, le=3)]


class Model(BaseModel):
    #              tensor type                          shape                    dtype
    tensor: Tensor[torch.Tensor | np.ndarray[Any, Any], tuple[DimType, DimType], Literal["int32", "int64"]]


parsed = Model.model_validate({"tensor": np.ones((2, 2), dtype="int32")})
# access the parsed tensor via the "value" property
parsed.tensor.value

# invalid shapes
Model.model_validate({"tensor": np.ones((1, 1), dtype="int32")})
Model.model_validate({"tensor": np.ones((4, 4), dtype="int32")})
Model.model_validate({"tensor": np.ones(2, dtype="int32")})
Model.model_validate({"tensor": np.ones((2, 2, 2), dtype="int32")})

# invalid dtype
Model.model_validate({"tensor": np.ones((2, 2), dtype="float32")})

# successfully validate np.ndarray
Model.model_validate({"tensor": np.ones((2, 2), dtype="int32")})
# convert tf.Tensor to torch.Tensor
Model.model_validate({"tensor": tf.ones((2, 2), dtype=tf.int32)})
```

### Parsing

The JSON representation of the tensor contains the:

- binary data of the tensor in little-endian format encoded in Base64
- shape of the tensor
- datatype of the tensor

```python
from typing import Any

import numpy as np
from pydantic import BaseModel

from pydantic_tensor import Tensor


class Model(BaseModel):
    tensor: Tensor[Any, Any, Any]


parsed = Model.model_validate({"tensor": np.ones((2, 2), dtype="float32")})
# parse to JSON: {"tensor":{"shape":[2,2],"dtype":"float32","data":"AACAPwAAgD8AAIA/AACAPw=="}}
json_dump = parsed.model_dump_json()
# parse back to tensor: array([[1., 1.], [1., 1.]], dtype=float32)
Model.model_validate_json(json_dump).tensor.value
```

### DType Collections

Types `Int, UInt, Float, Complex, BFloat` from `pydantic_tensor.types` are unions of dtypes according to their names.
For Example `Int` is defined as `Literal["int8", "int16", "int32", "int64"]`.

```python
from typing import Any

import numpy as np
from pydantic import BaseModel

from pydantic_tensor import Tensor
from pydantic_tensor.types import Int


class Model(BaseModel):
    tensor: Tensor[Any, Any, Int]


for dtype in ["int8", "int16", "int32", "int64"]:
    Model.model_validate({"tensor": np.ones((2, 2), dtype=dtype)})  # success

Model.model_validate({"tensor": np.ones((2, 2), dtype="float32")})  # failure
```

### Lazy Tensors

Use `JaxArray, NumpyNDArray, TensorflowTensor, TorchTensor` for lazy versions of tensors types.
They only handle tensors when their equivalent libraries (jax, numpy, tensorflow, torch) are imported somewhere else in the program.

```python
from typing import Any

import numpy as np
from pydantic import BaseModel

from pydantic_tensor import Tensor
from pydantic_tensor.backend.torch import TorchTensor


class Model(BaseModel):
    tensor: Tensor[TorchTensor, Any, Any]


Model.model_validate({"tensor": np.ones((2, 2), dtype="float32")})  # failure

import torch

Model.model_validate({"tensor": np.ones((2, 2), dtype="float32")})  # success
```

## Development

### Install pre-commit hooks

```
pre-commit install
```

### Lint

```
hatch run lint:all
```

### Test

```
hatch run test:test
```

### Check spelling

```
hatch run spell
```
