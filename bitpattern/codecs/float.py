"""IEEE 754 binary floats and similar formats as their raw bits."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from functools import cached_property

from ..intset import IntSet
from ..pattern import diagram
from .codec import BDDSet, OrderedCodec

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
class Float(OrderedCodec[float]):
    """An IEEE 754 binary format: a sign bit, then the exponent, then the mantissa.

    Sets iterate in bit order:
      - `+0.0` up to `inf`, then NaNs with the sign clear
      - `-0.0` down to `-inf`, then NaNs with the sign set
    `range`, `min` and `max` use numeric order.
    """

    exponent: int
    # For struct, eg. ">d". bfloat16 is the top half of a ">f" and float8_e5m2 the
    # top byte of an ">e".
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
        """`value` rounded to this format's precision, ties to even."""
        if not math.isfinite(value) or not value:
            return value
        # Normal floats have mantissa + 1 significant bits. Subnormals stop at
        # 2**_lowest. encode refuses anything too big to fit.
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
        """`value >= 0`, including `-0.0`."""
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
        """The key of the first float at or above `value`, where a range bounded by
        `value` starts or stops."""
        if value > self.max:  # inf if there is one, or else past the end
            return self._key(self.encode(self.max)) + 1
        if -math.inf < value < self.min:
            value = self.min
        key = self._key(bits := self.encode(value))
        # encode rounds to the nearest float, which can be the one below.
        return key + 1 if self.decode(bits) < value else key

    @property
    def min(self) -> float:
        """The most negative finite float, like `np.finfo`'s."""
        return (self.finite & self.negative)[-1]

    @property
    def max(self) -> float:
        """The largest finite float, like `np.finfo`'s."""
        return (self.finite & self.positive)[-1]

    def range(
        self, start: float | None = None, stop: float | None = None
    ) -> BDDSet[float]:
        """Floats in `[start, stop)`, never NaN. A bound of `None` includes the
        infinity at that end. `-0.0` sorts just below `0.0`, so `range(0.0)` leaves
        out `-0.0`."""
        if any(bound is not None and math.isnan(bound) for bound in (start, stop)):
            raise ValueError("NaN isn't ordered, so it can't bound a range")
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        low = self._bound(-math.inf if start is None else start)
        high = self._bound(self.positive[-1]) + 1 if stop is None else self._bound(stop)
        # Keys below sign are the negative floats (reversed) and the rest are positive.
        negative = IntSet.range(
            mask - min(high, sign) + 1, mask - low + 1, width=self.width
        )
        nonnegative = IntSet.range(max(low, sign) - sign, high - sign, width=self.width)
        return BDDSet(self, negative | nonnegative)


@dataclass(frozen=True, repr=False)
class FloatFN(Float):
    """A binary float with NaNs but no infinities, like float8_e4m3fn.

    FN is for finite and NaN. Only bit patterns with every exponent and mantissa bit
    set are NaN. Other patterns with every exponent bit set are normal numbers.
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
            # first exponent share that step. Each exponent after doubles the step.
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
        # With no -inf to encode, a range from -inf starts at the most negative float.
        return super()._bound(self.min if value == -math.inf else value)


float8_e5m2 = Float(8, "float8_e5m2", exponent=5, format=">e")
float8_e4m3fn = FloatFN(8, "float8_e4m3fn", exponent=4)
float16 = Float(16, "float16", exponent=5, format=">e")
bfloat16 = Float(16, "bfloat16", exponent=8, format=">f")
float32 = Float(32, "float32", exponent=8, format=">f")
float64 = Float(64, "float64", exponent=11, format=">d")
