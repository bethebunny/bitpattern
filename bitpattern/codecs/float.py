"""IEEE 754 binary floats, as their raw bits."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from functools import cached_property

from ..intset import IntSet
from ..pattern import build
from .codec import BDDSet, Codec

__all__ = ["Float", "float16", "float32", "float64"]


@dataclass(frozen=True, repr=False)
class Float(Codec[float]):
    """An IEEE 754 binary format: a sign bit, then the exponent, then the mantissa.

    Non-negative floats sort in the same order as their bits, so indexing them
    counts up from `+0.0`. Negative floats count down.
    """

    exponent: int
    format: str  # for struct, eg. ">d"

    @property
    def mantissa(self) -> int:
        return self.width - 1 - self.exponent

    def encode(self, value: object) -> int:
        try:
            return int.from_bytes(struct.pack(self.format, value))
        except (struct.error, OverflowError) as e:
            raise ValueError(f"{self} can't encode {value!r}") from e

    def decode(self, bits: int) -> float:
        return struct.unpack(self.format, bits.to_bytes(self.width // 8))[0]

    def _glob(self, *, sign: str, exponent: str, mantissa: str) -> BDDSet[float]:
        bits = sign + exponent * self.exponent + mantissa * self.mantissa
        return BDDSet(self, IntSet.from_bdd(build(bits), self.width))

    # The sign bit splits every bit pattern in half, NaNs included. positive and
    # negative are narrower, since a NaN has a sign but isn't positive or negative.

    @cached_property
    def sign_clear(self) -> BDDSet[float]:
        return self._glob(sign="0", exponent="?", mantissa="?")

    @cached_property
    def sign_set(self) -> BDDSet[float]:
        return self._glob(sign="1", exponent="?", mantissa="?")

    @cached_property
    def zeros(self) -> BDDSet[float]:
        """Both `-0.0` and `+0.0`."""
        return self._glob(sign="?", exponent="0", mantissa="0")

    @cached_property
    def positive(self) -> BDDSet[float]:
        return self.sign_clear - self.zeros - self.nan

    @cached_property
    def negative(self) -> BDDSet[float]:
        return self.sign_set - self.zeros - self.nan

    @cached_property
    def nonnegative(self) -> BDDSet[float]:
        """`value >= 0`, which `-0.0` is too."""
        return self.positive | self.zeros

    @cached_property
    def subnormal(self) -> BDDSet[float]:
        return self._glob(sign="?", exponent="0", mantissa="?") - self.zeros

    @cached_property
    def infinities(self) -> BDDSet[float]:
        return self._glob(sign="?", exponent="1", mantissa="0")

    @cached_property
    def nan(self) -> BDDSet[float]:
        return self._glob(sign="?", exponent="1", mantissa="?") - self.infinities

    @cached_property
    def finite(self) -> BDDSet[float]:
        return self.all - self._glob(sign="?", exponent="1", mantissa="?")

    def _key(self, bits: int) -> int:
        """Reorder bits so that unsigned order is float order. Negative floats get
        flipped, which reverses them, and positive floats get moved above them."""
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        return bits ^ mask if bits & sign else bits | sign

    def range(self, low: float, high: float) -> BDDSet[float]:
        """Floats in `[low, high)`."""
        if math.isnan(low) or math.isnan(high):
            raise ValueError("NaN isn't ordered, so it can't bound a range")
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        start, stop = self._key(self.encode(low)), self._key(self.encode(high))
        # Keys below sign are the negative floats, flipped, and the rest are positive.
        negative = IntSet.range(
            mask - min(stop, sign) + 1, mask - start + 1, width=self.width
        )
        nonnegative = IntSet.range(
            max(start, sign) - sign, stop - sign, width=self.width
        )
        return BDDSet(self, negative | nonnegative)


float16 = Float(16, "float16", exponent=5, format=">e")
float32 = Float(32, "float32", exponent=8, format=">f")
float64 = Float(64, "float64", exponent=11, format=">d")
