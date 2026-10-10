"""Fixed-width integers as their raw bits."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import SupportsIndex

from ..intset import IntSet
from .codec import BDDSet, OrderedCodec

__all__ = [
    "Int",
    "int8",
    "int16",
    "int32",
    "int64",
    "int128",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uint128",
]


@dataclass(frozen=True, repr=False)
class Int(OrderedCodec[int]):
    """Fixed-width integers, in two's complement when signed.

    Sets iterate in bit order, which for signed integers is 0 to `max`, then `min`
    to -1.
    """

    signed: bool

    @property
    def min(self) -> int:
        return -(1 << (self.width - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        return self.min + (1 << self.width) - 1

    def encode(self, value: object) -> int:
        if (
            isinstance(value, SupportsIndex)
            and self.min <= (integer := operator.index(value)) <= self.max
        ):
            return integer % (1 << self.width)
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> int:
        return bits - (1 << self.width) if bits > self.max else bits

    def range(self, start: int | None = None, stop: int | None = None) -> BDDSet[int]:
        """Integers in `[start, stop)` that fit this codec. A bound of `None` leaves
        that end open."""
        low = self.min if start is None else max(start, self.min)
        high = self.max + 1 if stop is None else min(stop, self.max + 1)
        # Negative integers are encoded above the rest, offset by 2**width.
        offset = 1 << self.width
        negative = IntSet.range(low + offset, min(high, 0) + offset, width=self.width)
        nonnegative = IntSet.range(max(low, 0), high, width=self.width)
        return BDDSet(self, negative | nonnegative)


int8 = Int(8, "int8", signed=True)
int16 = Int(16, "int16", signed=True)
int32 = Int(32, "int32", signed=True)
int64 = Int(64, "int64", signed=True)
int128 = Int(128, "int128", signed=True)
uint8 = Int(8, "uint8", signed=False)
uint16 = Int(16, "uint16", signed=False)
uint32 = Int(32, "uint32", signed=False)
uint64 = Int(64, "uint64", signed=False)
uint128 = Int(128, "uint128", signed=False)
