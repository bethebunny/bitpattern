"""IEEE 754 binary floats and formats like them, as their raw bits."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from functools import cached_property

from ..intset import IntSet
from ..pattern import diagram
from .codec import BDDSet, Codec

__all__ = [
    "Float",
    "FloatFN",
    "bfloat16",
    "float8_e4m3fn",
    "float8_e5m2",
    "float16",
    "float32",
    "float64",
]


@dataclass(frozen=True, repr=False)
class Float(Codec[float]):
    """An IEEE 754 binary format: a sign bit, then the exponent, then the mantissa.

    Non-negative floats sort in the same order as their bits, so indexing them
    counts up from `+0.0`. Negative floats count down.
    """

    exponent: int
    # For struct, eg. ">d". bfloat16 is the top half of a ">f", and float8_e5m2 is
    # the top byte of an ">e".
    format: str

    @property
    def mantissa(self) -> int:
        return self.width - 1 - self.exponent

    @cached_property
    def _spare(self) -> int:
        """How many bits `format` has below this format's."""
        return 8 * struct.calcsize(self.format) - self.width

    def encode(self, value: object) -> int:
        try:
            double = struct.unpack(">d", struct.pack(">d", value))[0]
            # Round to this format before packing. struct would round to the wider
            # format, and rounding twice can go the wrong way.
            packed = struct.pack(self.format, self._round(double))
        except (struct.error, OverflowError) as e:
            raise ValueError(f"{self} can't encode {value!r}") from e
        return int.from_bytes(packed) >> self._spare

    def decode(self, bits: int) -> float:
        packed = (bits << self._spare).to_bytes(struct.calcsize(self.format))
        return struct.unpack(self.format, packed)[0]

    @property
    def _lowest(self) -> int:
        """The exponent of the smallest subnormal, eg. -24 for a float16."""
        return 2 - (1 << (self.exponent - 1)) - self.mantissa

    def _round(self, value: float) -> float:
        """`value` rounded to this format's precision, with ties going to even."""
        if not math.isfinite(value) or not value:
            return value
        # Normal floats have mantissa + 1 significant bits, and subnormals stop at
        # 2**_lowest. Too big to fit is left for encode to refuse.
        step = max(math.frexp(value)[1] - self.mantissa - 1, self._lowest)
        return math.copysign(math.ldexp(round(math.ldexp(value, -step)), step), value)

    def _glob(self, *, sign: str, exponent: str, mantissa: str) -> BDDSet[float]:
        bits = sign + exponent * self.exponent + mantissa * self.mantissa
        return BDDSet(self, IntSet.from_bdd(diagram(bits), self.width))

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
        # Reorder bits so that unsigned order is float order.
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        return bits ^ mask if bits & sign else bits | sign

    def _bound(self, value: float) -> int:
        """The key where a range bounded by `value` starts or stops."""
        return self._key(self.encode(value))

    def range(self, low: float, high: float) -> BDDSet[float]:
        """Floats in `[low, high)`."""
        if math.isnan(low) or math.isnan(high):
            raise ValueError("NaN isn't ordered, so it can't bound a range")
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        start, stop = self._bound(low), self._bound(high)
        # Keys below sign are the negative floats (reversed) and the rest are positive.
        negative = IntSet.range(
            mask - min(stop, sign) + 1, mask - start + 1, width=self.width
        )
        nonnegative = IntSet.range(
            max(start, sign) - sign, stop - sign, width=self.width
        )
        return BDDSet(self, negative | nonnegative)


@dataclass(frozen=True, repr=False)
class FloatFN(Float):
    """A binary float with NaNs but no infinities, like float8_e4m3fn.

    FN is for finite and NaN. The patterns with every exponent bit set are normal
    numbers, except the ones with every mantissa bit set too, which are NaN. struct
    can't pack these, so encode and decode work out the bits themselves.
    """

    format: str = ""  # unused, since struct has no format for these

    @cached_property
    def _largest(self) -> float:
        return self.positive[-1]

    def encode(self, value: object) -> int:
        try:
            double = struct.unpack(">d", struct.pack(">d", value))[0]
            magnitude = abs(self._round(double))  # rounding up can overflow a double
        except (struct.error, OverflowError) as e:
            raise ValueError(f"{self} can't encode {value!r}") from e
        sign = 1 << (self.width - 1)
        if math.isnan(magnitude):
            bits = sign - 1
        elif magnitude > self._largest:
            raise ValueError(f"{self} can't encode {value!r}")
        else:
            # Values are whole steps of the smallest subnormal. Subnormals and the
            # first exponent share that step, and each exponent after doubles it.
            steps = int(math.ldexp(magnitude, -self._lowest))
            exponent = max(steps.bit_length() - self.mantissa, 0)
            mantissa = steps >> max(exponent - 1, 0)  # with the implicit leading 1
            bits = exponent << self.mantissa | mantissa % (1 << self.mantissa)
        return bits | sign if math.copysign(1, double) < 0 else bits

    def decode(self, bits: int) -> float:
        sign = 1 << (self.width - 1)
        if (magnitude := bits & (sign - 1)) == sign - 1:
            value = math.nan
        else:
            exponent, mantissa = divmod(magnitude, 1 << self.mantissa)
            if exponent:  # normal numbers have an implicit leading 1
                mantissa |= 1 << self.mantissa
            value = math.ldexp(mantissa, self._lowest + max(exponent - 1, 0))
        return -value if bits & sign else value

    @cached_property
    def infinities(self) -> BDDSet[float]:
        return self.none

    @cached_property
    def nan(self) -> BDDSet[float]:
        return self._glob(sign="?", exponent="1", mantissa="1")

    @cached_property
    def finite(self) -> BDDSet[float]:
        return self.all - self.nan

    def _bound(self, value: float) -> int:
        # There's no infinity to encode, but it still bounds a range. The NaNs have
        # the outermost keys, so -inf bounds just above the negative NaN, and +inf
        # at the positive NaN.
        if math.isinf(value):
            return 1 if value < 0 else (1 << self.width) - 1
        return super()._bound(value)


float8_e5m2 = Float(8, "float8_e5m2", exponent=5, format=">e")
float8_e4m3fn = FloatFN(8, "float8_e4m3fn", exponent=4)
float16 = Float(16, "float16", exponent=5, format=">e")
bfloat16 = Float(16, "bfloat16", exponent=8, format=">f")
float32 = Float(32, "float32", exponent=8, format=">f")
float64 = Float(64, "float64", exponent=11, format=">d")
