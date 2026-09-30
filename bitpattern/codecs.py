"""Floats and IP addresses as integers, so that sets of them are IntSets.

Values are encoded as their raw bits, so patterns match the bit layout. For
instance the float16 NaNs and infinities are `float16.pattern("?111.11??.*.*")`.
"""

from __future__ import annotations

import ipaddress
import math
import random
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

from .bdd import BDD, cofactors
from .pattern import Pattern, build
from .sets import IntSet

__all__ = ["IP", "Codec", "Float", "float16", "float32", "float64", "ipv4", "ipv6"]

T = TypeVar("T")
Address = TypeVar("Address", ipaddress.IPv4Address, ipaddress.IPv6Address)
Network = TypeVar("Network", ipaddress.IPv4Network, ipaddress.IPv6Network)


@dataclass(frozen=True, repr=False)
class Codec(Generic[T]):
    """A fixed-width encoding of values as non-negative integers."""

    width: int
    name: str

    def __repr__(self) -> str:
        return self.name

    def encode(self, value: T) -> int:
        raise NotImplementedError

    def decode(self, bits: int) -> T:
        raise NotImplementedError

    @cached_property
    def all(self) -> IntSet:
        """Every bit pattern of this width, whether or not it decodes to a value."""
        return IntSet.from_bdd(BDD.ACCEPT, self.width)

    @cached_property
    def none(self) -> IntSet:
        return IntSet.from_bdd(BDD.REJECT, self.width)

    def set(self, values: Iterable[T]) -> IntSet:
        return IntSet(map(self.encode, values), self.width)

    def pattern(self, text: str) -> Pattern:
        pattern = Pattern(text)
        if pattern.width != self.width:
            raise ValueError(f"pattern is {pattern.width} bits, not {self.width}")
        return pattern

    def values(self, source: IntSet) -> Iterator[T]:
        """Decode the members of `source`, in order of their bits."""
        return map(self.decode, source)

    def choice(
        self, source: IntSet | None = None, rng: random.Random | None = None
    ) -> T:
        """A random value from `source`, or from every bit pattern."""
        return self.decode((self.all if source is None else source).choice(rng))


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

    def encode(self, value: float) -> int:
        return int.from_bytes(struct.pack(self.format, value))

    def decode(self, bits: int) -> float:
        return struct.unpack(self.format, bits.to_bytes(self.width // 8))[0]

    def _field(self, sign: str, exponent: str, mantissa: str) -> IntSet:
        bits = sign + exponent * self.exponent + mantissa * self.mantissa
        return IntSet.from_bdd(build(bits), self.width)

    # The sign bit splits every bit pattern in half, NaNs included. positive and
    # negative are narrower, since a NaN has a sign but isn't positive or negative.

    @cached_property
    def sign_clear(self) -> IntSet:
        return self._field("0", "?", "?")

    @cached_property
    def sign_set(self) -> IntSet:
        return self._field("1", "?", "?")

    @cached_property
    def zeros(self) -> IntSet:
        """Both `-0.0` and `+0.0`."""
        return self._field("?", "0", "0")

    @cached_property
    def positive(self) -> IntSet:
        return self.sign_clear - self.zeros - self.nan

    @cached_property
    def negative(self) -> IntSet:
        return self.sign_set - self.zeros - self.nan

    @cached_property
    def nonnegative(self) -> IntSet:
        """`value >= 0`, which `-0.0` is too."""
        return self.positive | self.zeros

    @cached_property
    def subnormal(self) -> IntSet:
        return self._field("?", "0", "?") - self.zeros

    @cached_property
    def infinities(self) -> IntSet:
        return self._field("?", "1", "0")

    @cached_property
    def nan(self) -> IntSet:
        return self._field("?", "1", "?") - self.infinities

    @cached_property
    def finite(self) -> IntSet:
        return self.all - self._field("?", "1", "?")

    def _key(self, bits: int) -> int:
        """Reorder bits so that unsigned order is float order. Negative floats get
        flipped, which reverses them, and positive floats get moved above them."""
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        return bits ^ mask if bits & sign else bits | sign

    def range(self, low: float, high: float) -> IntSet:
        """Floats in `[low, high)`. NaNs aren't ordered, so they're never in a range."""
        if math.isnan(low) or math.isnan(high):
            raise ValueError("NaN isn't ordered, so it can't bound a range")
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        start, stop = self._key(self.encode(low)), self._key(self.encode(high))
        # Keys below sign are the negative floats, flipped, and the rest are positive.
        negative = IntSet.range(
            mask - min(stop, sign) + 1, mask - start + 1, self.width
        )
        nonnegative = IntSet.range(max(start, sign) - sign, stop - sign, self.width)
        return negative | nonnegative


@dataclass(frozen=True, repr=False)
class IP(Codec[Address], Generic[Address, Network]):
    """IPv4 or IPv6 addresses, which are integers already."""

    # The family's own classes, since ipaddress.ip_address() guesses the family
    # from the value, and would decode every IPv6 address below 2**32 as IPv4.
    address_type: type[Address]
    network_type: type[Network]

    def encode(self, value: Address | str | int) -> int:
        return int(self.address_type(value))

    def decode(self, bits: int) -> Address:
        return self.address_type(bits)

    def cidr(self, network: Network | str) -> IntSet:
        block = self.network_type(network, strict=False)
        start = int(block.network_address)
        return IntSet.range(start, start + block.num_addresses, self.width)

    def range(self, low: Address | str | int, high: Address | str | int) -> IntSet:
        """Addresses in `[low, high)`."""
        return IntSet.range(self.encode(low), self.encode(high), self.width)

    def networks(self, source: IntSet) -> Iterator[Network]:
        """The fewest CIDR blocks that make up `source`, ie. route aggregation.

        Each path to ACCEPT is a block, and since the diagram is reduced, each one
        is as big as it can be. Unlike a pattern, a block's free bits all have to
        be at the end, so this splits even the bits that the diagram doesn't test.
        """

        def walk(node: BDD, bit: int, prefix: int) -> Iterator[Network]:
            if node is BDD.ACCEPT:
                yield self.network_type((prefix << (bit + 1), self.width - 1 - bit))
            elif node:
                low, high = cofactors(node, bit)
                yield from walk(low, bit - 1, prefix << 1)
                yield from walk(high, bit - 1, prefix << 1 | 1)

        if source and source[-1].bit_length() > self.width:
            raise ValueError(f"set has members wider than {self.width} bits")
        # A wider set's extra high bits are all clear, so walking them adds nothing.
        wide = source.widen(self.width)
        yield from walk(wide.bdd, wide.width - 1, 0)


float16 = Float(16, "float16", exponent=5, format=">e")
float32 = Float(32, "float32", exponent=8, format=">f")
float64 = Float(64, "float64", exponent=11, format=">d")

ipv4 = IP(32, "ipv4", ipaddress.IPv4Address, ipaddress.IPv4Network)
ipv6 = IP(128, "ipv6", ipaddress.IPv6Address, ipaddress.IPv6Network)
