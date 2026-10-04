"""Codecs for floats, integers and IP addresses, and BDDSets, the sets of values they
make.

A BDDSet keeps its values as the IntSet of their encodings, which are the values'
raw bits, so patterns match the bit layout. For instance the float16 NaNs and
infinities are `float16.pattern("?111.11??.*.*")`.
"""

from .codec import BDDSet, Codec
from .float import Float, bfloat16, float16, float32, float64
from .int import (
    Int,
    int8,
    int16,
    int32,
    int64,
    int128,
    uint8,
    uint16,
    uint32,
    uint64,
    uint128,
)
from .ip import IP, ipv4, ipv6
from .tuple import Tuple

__all__ = [
    "IP",
    "BDDSet",
    "Codec",
    "Float",
    "Int",
    "Tuple",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "int128",
    "ipv4",
    "ipv6",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uint128",
]
