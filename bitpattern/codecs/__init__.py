"""Codecs for floats and IP addresses, and BDDSets, the sets of values they make.

A BDDSet keeps its values as the IntSet of their encodings, which are the values'
raw bits, so patterns match the bit layout. For instance the float16 NaNs and
infinities are `float16.pattern("?111.11??.*.*")`.
"""

from .codec import BDDSet, Codec
from .float import Float, float16, float32, float64
from .ip import IP, ipv4, ipv6

__all__ = [
    "IP",
    "BDDSet",
    "Codec",
    "Float",
    "float16",
    "float32",
    "float64",
    "ipv4",
    "ipv6",
]
