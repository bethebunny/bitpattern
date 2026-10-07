"""Codecs for floats, integers and IP addresses, and BDDSets, the sets of values they
make.

A BDDSet keeps its values as the IntSet of their encodings, which are the values'
raw bits, so patterns match the bit layout. For instance the float16 NaNs and
infinities are `float16.pattern("?111.11??.*.*")`.
"""

from .char import Char, char
from .codec import BDDSet, Codec, Tuple, product, project
from .enum import Flags, Members
from .float import (
    Float,
    FloatFN,
    bfloat16,
    float8_e4m3fn,
    float8_e5m2,
    float16,
    float32,
    float64,
)
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
from .mac import MAC, mac
from .time import Date, Datetime, dates, datetimes
from .uuid import UUIDCodec, uuid128

__all__ = [
    "IP",
    "MAC",
    "BDDSet",
    "Char",
    "Codec",
    "Date",
    "Datetime",
    "Flags",
    "Float",
    "FloatFN",
    "Int",
    "Members",
    "Tuple",
    "UUIDCodec",
    "bfloat16",
    "char",
    "dates",
    "datetimes",
    "float8_e4m3fn",
    "float8_e5m2",
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
    "mac",
    "product",
    "project",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uint128",
    "uuid128",
]
