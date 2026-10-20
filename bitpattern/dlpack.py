"""Sets' members as DLPack tensors, eg. for `numpy.from_dlpack`.

https://dmlc.github.io/dlpack/latest/python_spec.html
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from .bdd import BDD, tensor

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

__all__ = ["DType"]

CPU = 1  # DLPack's device type for host memory

# DLPack's type codes for ints.
INT = 0
UINT = 1


class DType(Enum):
    """A C type that sets can export their values as. Values are DLPack's
    `(code, bits)`, and names are NumPy's (or ml_dtypes' for bfloat16 and float8s)."""

    INT8 = (INT, 8)
    INT16 = (INT, 16)
    INT32 = (INT, 32)
    INT64 = (INT, 64)
    UINT8 = (UINT, 8)
    UINT16 = (UINT, 16)
    UINT32 = (UINT, 32)
    UINT64 = (UINT, 64)
    FLOAT16 = (2, 16)
    FLOAT32 = (2, 32)
    FLOAT64 = (2, 64)
    BFLOAT16 = (4, 16)
    FLOAT8_E4M3FN = (10, 8)
    FLOAT8_E5M2 = (12, 8)

    @property
    def code(self) -> int:
        return self.value[0]

    @property
    def bits(self) -> int:
        return self.value[1]


FROM_DLPACK_1_1 = frozenset({DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2})


def unsigned(width: int) -> DType:
    """The narrowest unsigned int that fits, or rows of UINT8 past 64 bits."""
    types = (DType.UINT8, DType.UINT16, DType.UINT32, DType.UINT64)
    return next((dtype for dtype in types if width <= dtype.bits), DType.UINT8)


def signed(width: int) -> DType:
    """For widths up to 64."""
    return DType((INT, unsigned(width).bits))


def exported(
    node: BDD,
    *,
    width: int,
    dtype: DType,
    stream: int | None,
    max_version: tuple[int, int] | None,
    dl_device: tuple[int, int] | None,
    copy: bool | None,
) -> CapsuleType:
    """`__dlpack__` for `node`'s members below `2 ** width`."""
    if (dtype.code == INT and width > dtype.bits) or (
        dtype.code > UINT and width != dtype.bits
    ):
        raise ValueError(f"{dtype.name.lower()} can't hold {width}-bit values")
    if stream is not None:
        raise ValueError("sets are in host memory, which takes no stream")
    if dl_device is not None and tuple(dl_device) != (CPU, 0):
        raise BufferError(f"sets are in host memory, not on device {dl_device}")
    if copy is False:
        raise BufferError("sets have no array to share, so they're exported as copies")
    # Minor versions only add type codes, so float8s go out as 1.1 even to consumers
    # that ask for 1.0 (eg. torch, which reads them).
    if max_version is not None and tuple(max_version) >= (1, 0):
        version = (1, 1) if dtype in FROM_DLPACK_1_1 else (1, 0)
    else:
        version = None  # a consumer from before DLPack 1.0
    return tensor(node, width=width, code=dtype.code, bits=dtype.bits, version=version)
