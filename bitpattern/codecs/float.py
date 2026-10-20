"""IEEE 754 binary floats and similar formats as their raw bits."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from functools import cached_property

from ..bdd import diagram
from ..dlpack import DType, unsigned
from ..intset import IntSet
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

# Formats whose encodings are a C type's bits, by (width, exponent, struct format).
FLOAT_DTYPES = {
    (8, 5, ">e"): DType.FLOAT8_E5M2,
    (16, 5, ">e"): DType.FLOAT16,
    (16, 8, ">f"): DType.BFLOAT16,
    (32, 8, ">f"): DType.FLOAT32,
    (64, 11, ">d"): DType.FLOAT64,
}


@dataclass(frozen=True, repr=False)
class Float(OrderedCodec[float]):
    """An IEEE 754 binary format: sign :: exponent :: mantissa

    Sets iterate in bit order:
      - `+0.0` up to `inf`, then NaNs with the sign clear
      - `-0.0` down to `-inf`, then NaNs with the sign set
    `range`, `min` and `max` use numeric order.
    """

    exponent: int
    format: str  # `struct` format, eg. ">d"

    @cached_property
    def mantissa(self) -> int:
        return self.width - 1 - self.exponent

    # struct caches compiled formats, and a codec holding one couldn't be pickled.
    @cached_property
    def _bytes(self) -> int:
        """How many bytes `format` packs a float into."""
        return struct.calcsize(self.format)

    @cached_property
    def _spare(self) -> int:
        """How many bits `format` has below this format's."""
        return 8 * self._bytes - self.width

    def _double(self, value: object) -> float:
        try:
            if isinstance(value, int):  # struct would call one too big a non-number
                return float(value)
            return struct.unpack(">d", struct.pack(">d", value))[0]
        except struct.error as e:  # not a number
            raise TypeError(f"{self} can't encode {value!r}") from e
        except OverflowError as e:  # an int too big for a double
            raise ValueError(f"{self} can't encode {value!r}") from e

    def encode(self, value: object) -> int:
        double = value if type(value) is float else self._double(value)
        try:
            # A narrower format rounds to its own precision first, since struct's
            # rounding to `format` on top could round twice the wrong way.
            rounded = self._round(double) if self._spare else double
            packed = struct.pack(self.format, rounded)
        except OverflowError as e:
            raise ValueError(f"{self} can't encode {value!r}") from e
        return int.from_bytes(packed) >> self._spare

    def decode(self, bits: int) -> float:
        try:
            packed = (bits << self._spare).to_bytes(self._bytes)
        except OverflowError as e:  # wider than this format
            raise IndexError(f"{self} has no value encoded as {bits}") from e
        return struct.unpack(self.format, packed)[0]

    def dtype(self) -> DType:
        key = (self.width, self.exponent, self.format)
        return FLOAT_DTYPES[key] if key in FLOAT_DTYPES else super().dtype()

    @cached_property
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
        """The key of the first float at or above `value`."""
        if value > self.max:  # inf if there is one, or else past the end
            return self._key(self.encode(self.max)) + 1
        if -math.inf < value < self.min:
            value = self.min
        key = self._key(bits := self.encode(value))
        # encode rounds to the nearest float, which can be the one below.
        return key + 1 if self.decode(bits) < value else key

    @cached_property
    def min(self) -> float:
        """The most negative finite float, like `np.finfo`'s."""
        return (self.finite & self.negative)[-1]

    @cached_property
    def max(self) -> float:
        """The largest finite float, like `np.finfo`'s."""
        return (self.finite & self.positive)[-1]

    def range(
        self, start: float | None = None, stop: float | None = None
    ) -> BDDSet[float]:
        """Floats in `[start, stop)`, never NaN. `None` includes that end's infinity.
        `-0.0` sorts just below `0.0`, so `range(0.0)` leaves it out."""
        if any(bound is not None and math.isnan(bound) for bound in (start, stop)):
            raise ValueError("NaN isn't ordered, so it can't bound a range")
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        low = self._bound(-math.inf if start is None else start)
        high = self._bound(self.positive[-1]) + 1 if stop is None else self._bound(stop)
        # Keys below `sign` are the negative floats, reversed.
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

    def dtype(self) -> DType:
        if (self.width, self.exponent) == (8, 4):
            return DType.FLOAT8_E4M3FN
        # Float's dtypes have infinities where these formats have numbers.
        return unsigned(self.width)

    @cached_property
    def _largest(self) -> float:
        return self.positive[-1]

    def encode(self, value: object) -> int:
        double = value if type(value) is float else self._double(value)
        try:
            magnitude = abs(self._round(double))  # rounding up can overflow a double
        except OverflowError as e:
            raise ValueError(f"{self} can't encode {value!r}") from e
        sign = 1 << (self.width - 1)
        if math.isnan(magnitude):
            bits = sign - 1
        elif magnitude > self._largest:
            raise ValueError(f"{self} can't encode {value!r}")
        else:
            # Whole steps of the smallest subnormal, which double with each exponent
            # after the first.
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
