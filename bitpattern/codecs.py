"""Sets of floats, IP addresses and the like, over their raw bit patterns.

Raw rather than rearranged into value order, so that `Pattern("?111.1111.1111.…")`
names the float64 NaNs because that is what their bits look like.
"""

from __future__ import annotations

import ipaddress
import math
import random
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .bdd import ACCEPT, REJECT, cofactors
from .pattern import Pattern, build
from .sets import IntSet

__all__ = ["Codec", "Float", "IP", "float16", "float32", "float64", "ipv4", "ipv6"]


@dataclass(frozen=True, repr=False)
class Codec:
    """A fixed-width encoding of some value type as a non-negative integer."""

    width: int
    name: str

    def __repr__(self) -> str:
        return self.name

    def encode(self, value: Any) -> int:
        raise NotImplementedError

    def decode(self, bits: int) -> Any:
        raise NotImplementedError

    @property
    def all(self) -> IntSet:
        """Every bit pattern of this width, whether or not it encodes a value."""
        return IntSet.from_bdd(ACCEPT, self.width)

    @property
    def none(self) -> IntSet:
        return IntSet.from_bdd(REJECT, self.width)

    def set(self, values: Iterable[Any]) -> IntSet:
        return IntSet(map(self.encode, values), self.width)

    def pattern(self, text: str) -> Pattern:
        parsed = Pattern(text)
        if parsed.width != self.width:
            raise ValueError(f"pattern is {parsed.width} bits wide, not {self.width}")
        return parsed

    def values(self, source: IntSet) -> Iterator[Any]:
        """Decode a set's members, in ascending bit order."""
        return map(self.decode, source)

    def sample(self, source: IntSet | None = None, rng: random.Random | None = None) -> Any:
        return self.decode((self.all if source is None else source).sample(rng))


@dataclass(frozen=True, repr=False)
class Float(Codec):
    """An IEEE-754 binary format: sign, then exponent, then mantissa bits.

    For non-negative floats, bit order is value order, so indexing them walks up
    in magnitude from `+0.0`. Negative floats run the other way.
    """

    exponent: int
    format: str

    @property
    def mantissa(self) -> int:
        return self.width - self.exponent - 1

    def encode(self, value: float) -> int:
        return int.from_bytes(struct.pack(self.format, value), "big")

    def decode(self, bits: int) -> float:
        return struct.unpack(self.format, bits.to_bytes(self.width // 8, "big"))[0]

    def _field(self, sign: str, exponent: str, mantissa: str) -> IntSet:
        return IntSet.from_bdd(build(sign + exponent * self.exponent + mantissa * self.mantissa), self.width)

    # The sign bit halves the universe, NaNs included. The value-level sets below
    # are narrower, because a NaN has a sign but is neither positive nor negative.

    @property
    def sign_clear(self) -> IntSet:
        return self._field("0", "?", "?")

    @property
    def sign_set(self) -> IntSet:
        return self._field("1", "?", "?")

    @property
    def zeros(self) -> IntSet:
        """Both `-0.0` and `+0.0`."""
        return self._field("?", "0", "0")

    @property
    def positive(self) -> IntSet:
        return self.sign_clear - self.zeros - self.nan

    @property
    def negative(self) -> IntSet:
        return self.sign_set - self.zeros - self.nan

    @property
    def nonnegative(self) -> IntSet:
        """`value >= 0`, which `-0.0` satisfies too."""
        return self.positive | self.zeros

    @property
    def subnormal(self) -> IntSet:
        return self._field("?", "0", "?") - self.zeros

    @property
    def infinities(self) -> IntSet:
        return self._field("?", "1", "0")

    @property
    def nan(self) -> IntSet:
        return self._field("?", "1", "?") - self.infinities

    @property
    def finite(self) -> IntSet:
        return self.all - self._field("?", "1", "?")

    def _key(self, bits: int) -> int:
        """Reorder a bit pattern so unsigned order is float order: negatives are
        flipped to reverse them below the positives, which are lifted above."""
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        return bits ^ mask if bits & sign else bits | sign

    def range(self, low: float, high: float) -> IntSet:
        """Bit patterns for the values in `[low, high)`. NaNs sort outside every
        finite range, so finite bounds exclude them."""
        if math.isnan(low) or math.isnan(high):
            raise ValueError("NaN is not ordered, so it cannot bound a range")
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        start, stop = self._key(self.encode(low)), self._key(self.encode(high))
        # Keys below `sign` are the negatives, reversed; the rest are shifted.
        negative = IntSet.range(mask - min(stop, sign) + 1, mask - start + 1, self.width)
        nonnegative = IntSet.range(max(start, sign) - sign, stop - sign, self.width)
        return negative | nonnegative


@dataclass(frozen=True, repr=False)
class IP(Codec):
    """IPv4 or IPv6 addresses, whose encoding is already the natural one."""

    # The family's own classes, not `ipaddress.ip_address`, which guesses the
    # family from the value and so decodes every IPv6 address below 2**32 as IPv4.
    address_type: type[ipaddress.IPv4Address] | type[ipaddress.IPv6Address]
    network_type: type[ipaddress.IPv4Network] | type[ipaddress.IPv6Network]

    def encode(self, value: Any) -> int:
        return int(self.address_type(value))

    def decode(self, bits: int) -> Any:
        return self.address_type(bits)

    def cidr(self, text: Any) -> IntSet:
        network = self.network_type(text, strict=False)
        start = int(network.network_address)
        return IntSet.range(start, start + network.num_addresses, self.width)

    def range(self, low: Any, high: Any) -> IntSet:
        """Addresses in `[low, high)`."""
        return IntSet.range(self.encode(low), self.encode(high), self.width)

    def networks(self, source: IntSet) -> Iterator[Any]:
        """The minimal CIDR cover of a set: route aggregation, read off the diagram.

        Each `ACCEPT` is one block, and reduction makes each as large as it can be.
        Unlike `Pattern` branches this cover is minimal, since prefixes nest.
        """
        def walk(node, top: int, prefix: int) -> Iterator[Any]:
            if node is REJECT:
                return
            if node is ACCEPT:
                yield self.network_type((prefix << (top + 1), self.width - 1 - top))
                return
            # Split even a free bit: a block has to be a prefix.
            low, high = cofactors(node, top)
            yield from walk(low, top - 1, prefix << 1)
            yield from walk(high, top - 1, prefix << 1 | 1)

        bdd, width = source._canonical()
        if width > self.width:
            raise ValueError(f"set has members wider than {self.width} bits")
        yield from walk(IntSet.from_bdd(bdd, width).widen(self.width).bdd, self.width - 1, 0)


float16 = Float(16, "float16", exponent=5, format=">e")
float32 = Float(32, "float32", exponent=8, format=">f")
float64 = Float(64, "float64", exponent=11, format=">d")

ipv4 = IP(32, "ipv4", ipaddress.IPv4Address, ipaddress.IPv4Network)
ipv6 = IP(128, "ipv6", ipaddress.IPv6Address, ipaddress.IPv6Network)
