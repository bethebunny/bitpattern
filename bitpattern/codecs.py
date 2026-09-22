"""Value codecs: sets of floats, IP addresses and the like, over raw bit patterns.

A codec pairs a width with a value/bits bijection, and hands back `IntSet`s over
the *raw* encoding rather than some order-preserving rearrangement of it. That is
the whole point of the library: `Pattern("?111.1111.1111.…")` should name the
NaNs of a float64 because that is genuinely what their bits look like. Where a
raw encoding is not monotonic in value -- IEEE-754 is sign-magnitude, so negative
floats run backwards -- `range` recovers the ordering per half instead.
"""

from __future__ import annotations

import ipaddress
import random
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .bdd import ACCEPT, REJECT
from .sets import IntSet
from .pattern import Pattern, build

__all__ = [
    "Codec",
    "Float",
    "IP",
    "float16",
    "float32",
    "float64",
    "ipv4",
    "ipv6",
]


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
        """Every bit pattern of this width, whether or not it is a valid value."""
        return IntSet.from_bdd(ACCEPT, self.width)

    @property
    def none(self) -> IntSet:
        return IntSet.from_bdd(REJECT, self.width)

    def set(self, values: Iterable[Any]) -> IntSet:
        return IntSet((self.encode(value) for value in values), self.width)

    def pattern(self, text: str) -> Pattern:
        """A `Pattern` over this codec's bits, checked against its width."""
        parsed = Pattern(text)
        if parsed.width != self.width:
            raise ValueError(f"pattern is {parsed.width} bits wide, not {self.width}")
        return parsed

    def values(self, source: IntSet) -> Iterator[Any]:
        """Decode a set's members, in the set's own ascending bit order."""
        return (self.decode(bits) for bits in source)

    def sample(self, source: IntSet | None = None, rng=random) -> Any:
        """One uniformly drawn value, decoded."""
        return self.decode((self.all if source is None else source).sample(rng))


@dataclass(frozen=True, repr=False)
class Float(Codec):
    """An IEEE-754 binary interchange format.

    Bits run sign, then exponent, then mantissa, so for non-negative floats the
    bit-pattern order *is* the value order -- which is why indexing a set of
    positive floats walks them in increasing magnitude, starting at `+0.0`.
    Negative floats run the other way; `range` accounts for that.
    """

    exponent: int
    format: str

    @property
    def mantissa(self) -> int:
        return self.width - self.exponent - 1

    @property
    def _sign(self) -> int:
        """The sign bit, as a mask."""
        return 1 << (self.width - 1)

    @property
    def _mask(self) -> int:
        return (1 << self.width) - 1

    def encode(self, value: float) -> int:
        return int.from_bytes(struct.pack(self.format, value), "big")

    def decode(self, bits: int) -> float:
        return struct.unpack(self.format, bits.to_bytes(self.width // 8, "big"))[0]

    # -- named sets, each just a constraint on the exponent and mantissa fields --

    def _field(self, sign: str, exponent: str, mantissa: str) -> IntSet:
        bits = sign + exponent * self.exponent + mantissa * self.mantissa
        return IntSet.from_bdd(build(bits), self.width)

    # The sign bit halves the universe; the value-level sets below are narrower,
    # because a NaN has a sign bit but is neither positive nor negative.

    @property
    def sign_clear(self) -> IntSet:
        """Sign bit clear: `+0.0` up to `+inf`, and the positive NaNs."""
        return self._field("0", "?", "?")

    @property
    def sign_set(self) -> IntSet:
        """Sign bit set: `-0.0` down to `-inf`, and the negative NaNs."""
        return self._field("1", "?", "?")

    @property
    def zeros(self) -> IntSet:
        """Both `-0.0` and `+0.0` -- everything for which `value == 0`."""
        return self._field("?", "0", "0")

    @property
    def positive(self) -> IntSet:
        """Everything for which `value > 0`, so neither zero nor a NaN."""
        return self.sign_clear - self.zeros - self.nan

    @property
    def negative(self) -> IntSet:
        """Everything for which `value < 0`, so neither zero nor a NaN."""
        return self.sign_set - self.zeros - self.nan

    @property
    def nonnegative(self) -> IntSet:
        """Everything for which `value >= 0` -- which `-0.0` satisfies too."""
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

    # -- ordering ------------------------------------------------------------

    def _key(self, bits: int) -> int:
        """Reorder a bit pattern so that unsigned order matches float order.

        Flipping every bit of a negative reverses the sign-magnitude run and
        puts it below the positives; setting the sign bit on a non-negative
        lifts it above them.
        """
        return bits ^ self._mask if bits & self._sign else bits | self._sign

    def range(self, low: float, high: float) -> IntSet:
        """Bit patterns for the values in `[low, high)`, in *value* order.

        NaNs sort outside the finite range at both ends, so finite bounds
        exclude them without having to say so.
        """
        for bound in (low, high):
            if bound != bound:
                raise ValueError("NaN is not ordered, so it cannot bound a range")
        start, stop = self._key(self.encode(low)), self._key(self.encode(high))
        if start >= stop:
            return self.none
        # Below the sign bit lie the negatives, reversed; above it the rest.
        negative = IntSet.range(
            self._mask - min(stop, self._sign) + 1, self._mask - start + 1, self.width
        ) if start < self._sign else self.none
        nonnegative = IntSet.range(
            max(start, self._sign) - self._sign, stop - self._sign, self.width
        ) if stop > self._sign else self.none
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

    def network(self, text: Any) -> Any:
        return self.network_type(text, strict=False)

    def cidr(self, text: Any) -> IntSet:
        """The addresses in one CIDR block."""
        network = self.network(text)
        start = int(network.network_address)
        return IntSet.range(start, start + network.num_addresses, self.width)

    def range(self, low: Any, high: Any) -> IntSet:
        """Addresses in `[low, high)`."""
        return IntSet.range(self.encode(low), self.encode(high), self.width)

    def networks(self, source: IntSet) -> Iterator[Any]:
        """The minimal CIDR cover of a set, read straight off the diagram.

        Every `ACCEPT` subtree is a whole block, and reduction guarantees those
        subtrees are as large as they can be -- so walking the diagram and
        emitting one network per `ACCEPT` yields the cover, which is exactly
        route aggregation. Minimal here is literal, unlike for `Pattern`
        branches: prefixes nest, so any cover by prefixes merges upward into
        this one.
        """
        def walk(node, top: int, prefix: int) -> Iterator[Any]:
            if node is REJECT:
                return
            if node is ACCEPT:
                yield self.network_type((prefix << (top + 1), self.width - 1 - top))
                return
            low, high = (node, node) if node.bit < top else (node.left, node.right)
            yield from walk(low, top - 1, prefix << 1)
            yield from walk(high, top - 1, (prefix << 1) | 1)

        bdd, width = source._canonical()
        if width > self.width:
            raise ValueError(f"set has members wider than {self.width} bits")
        yield from walk(IntSet.from_bdd(bdd, width).widen(self.width).bdd, self.width - 1, 0)


float16 = Float(width=16, name="float16", exponent=5, format=">e")
float32 = Float(width=32, name="float32", exponent=8, format=">f")
float64 = Float(width=64, name="float64", exponent=11, format=">d")

ipv4 = IP(32, "ipv4", ipaddress.IPv4Address, ipaddress.IPv4Network)
ipv6 = IP(128, "ipv6", ipaddress.IPv6Address, ipaddress.IPv6Network)
