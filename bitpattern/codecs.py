"""Codecs for floats and IP addresses, and BDDSets, the sets of values they make.

A BDDSet keeps its values as the IntSet of their encodings, which are the values'
raw bits, so patterns match the bit layout. For instance the float16 NaNs and
infinities are `float16.pattern("?111.11??.*.*")`.
"""

from __future__ import annotations

import ipaddress
import itertools
import math
import random
import struct
from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar, overload

from .bdd import BDD, cofactors, render_count
from .pattern import Pattern, build
from .sets import IntSet

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

    def encode(self, value: object) -> int:
        """`value`'s bits. Raises ValueError if this codec can't encode it."""
        raise NotImplementedError

    def decode(self, bits: int) -> T:
        raise NotImplementedError

    @cached_property
    def all(self) -> BDDSet[T]:
        """Every bit pattern of this width, whether or not it decodes to a value."""
        return BDDSet(self, IntSet.from_bdd(BDD.ACCEPT, self.width))

    @cached_property
    def none(self) -> BDDSet[T]:
        return BDDSet(self, IntSet.from_bdd(BDD.REJECT, self.width))

    def set(self, values: Iterable[T]) -> BDDSet[T]:
        return BDDSet(self, IntSet(map(self.encode, values), width=self.width))

    def pattern(self, text: str) -> BDDSet[T]:
        pattern = Pattern(text)
        if pattern.width != self.width:
            raise ValueError(f"pattern is {pattern.width} bits, not {self.width}")
        return BDDSet(self, pattern)


@dataclass(frozen=True, repr=False)
class BDDSet(AbstractSet[T], Sequence[T], Generic[T]):
    """A set of a codec's values, kept as the IntSet of their bits.

    It's also a sequence of the values, in the order of their bits. Set operations
    work on the bits, so none of them enumerate anything.
    """

    codec: Codec[T]
    bits: IntSet

    def __post_init__(self) -> None:
        if self.bits.width != self.codec.width:
            raise ValueError(
                f"{self.codec} needs {self.codec.width} bits, not {self.bits.width}"
            )

    def _bits(self, other: object) -> IntSet:
        # Raise rather than return NotImplemented, since IntSet's reflected
        # operators would try to enumerate the set.
        if not isinstance(other, BDDSet) or other.codec != self.codec:
            raise TypeError(f"{self.codec} sets only combine with each other")
        return other.bits

    @property
    def size(self) -> int:
        return self.bits.size

    def __len__(self) -> int:
        return len(self.bits)

    def __bool__(self) -> bool:
        return bool(self.bits)

    def __contains__(self, value: object) -> bool:
        try:
            return self.codec.encode(value) in self.bits
        except ValueError:  # not something this codec can encode
            return False

    def __iter__(self) -> Iterator[T]:
        return map(self.codec.decode, self.bits)

    def __reversed__(self) -> Iterator[T]:
        return map(self.codec.decode, reversed(self.bits))

    @overload
    def __getitem__(self, item: int) -> T: ...
    @overload
    def __getitem__(self, item: slice) -> BDDSet[T]: ...
    def __getitem__(self, item: int | slice) -> T | BDDSet[T]:
        if isinstance(item, slice):
            return BDDSet(self.codec, self.bits[item])
        return self.codec.decode(self.bits[item])

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        try:
            return self.bits.index(self.codec.encode(value), start, stop)
        except ValueError:
            raise ValueError(f"{value!r} is not in the set") from None

    def count(self, value: object) -> int:
        return int(value in self)

    def choice(self, *, rng: random.Random | None = None) -> T:
        """A uniformly random value."""
        return self.codec.decode(self.bits.choice(rng=rng))

    def __and__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self.bits & self._bits(other))

    def __or__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self.bits | self._bits(other))

    def __sub__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self.bits - self._bits(other))

    def __xor__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self.bits ^ self._bits(other))

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    def __rsub__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self._bits(other) - self.bits)

    def __invert__(self) -> BDDSet[T]:
        """Every other bit pattern this codec has."""
        return BDDSet(self.codec, ~self.bits)

    def __le__(self, other: object) -> bool:
        return self.bits <= self._bits(other)

    def __lt__(self, other: object) -> bool:
        return self.bits < self._bits(other)

    def __ge__(self, other: object) -> bool:
        return self.bits >= self._bits(other)

    def __gt__(self, other: object) -> bool:
        return self.bits > self._bits(other)

    def isdisjoint(self, other: Iterable[T]) -> bool:
        return self.bits.isdisjoint(self._bits(other))

    def __repr__(self) -> str:
        # Like IntSet's, but with the codec, and values rather than bits.
        if (size := self.size) <= 10:
            return f"BDDSet({self.codec}, {list(self)})"
        head = ", ".join(map(repr, itertools.islice(self, 4)))
        return (
            f"BDDSet({self.codec}, [{head}, ..., {self[-2]!r}, {self[-1]!r}], "
            f"size={render_count(size)})"
        )


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

    def _field(self, sign: str, exponent: str, mantissa: str) -> BDDSet[float]:
        bits = sign + exponent * self.exponent + mantissa * self.mantissa
        return BDDSet(self, IntSet.from_bdd(build(bits), self.width))

    # The sign bit splits every bit pattern in half, NaNs included. positive and
    # negative are narrower, since a NaN has a sign but isn't positive or negative.

    @cached_property
    def sign_clear(self) -> BDDSet[float]:
        return self._field("0", "?", "?")

    @cached_property
    def sign_set(self) -> BDDSet[float]:
        return self._field("1", "?", "?")

    @cached_property
    def zeros(self) -> BDDSet[float]:
        """Both `-0.0` and `+0.0`."""
        return self._field("?", "0", "0")

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
        return self._field("?", "0", "?") - self.zeros

    @cached_property
    def infinities(self) -> BDDSet[float]:
        return self._field("?", "1", "0")

    @cached_property
    def nan(self) -> BDDSet[float]:
        return self._field("?", "1", "?") - self.infinities

    @cached_property
    def finite(self) -> BDDSet[float]:
        return self.all - self._field("?", "1", "?")

    def _key(self, bits: int) -> int:
        """Reorder bits so that unsigned order is float order. Negative floats get
        flipped, which reverses them, and positive floats get moved above them."""
        sign, mask = 1 << (self.width - 1), (1 << self.width) - 1
        return bits ^ mask if bits & sign else bits | sign

    def range(self, low: float, high: float) -> BDDSet[float]:
        """Floats in `[low, high)`. NaNs aren't ordered, so they're never in a range."""
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


@dataclass(frozen=True, repr=False)
class IP(Codec[Address], Generic[Address, Network]):
    """IPv4 or IPv6 addresses, which are integers already."""

    # The family's own classes, since ipaddress.ip_address() guesses the family
    # from the value, and would decode every IPv6 address below 2**32 as IPv4.
    address_type: type[Address]
    network_type: type[Network]

    def encode(self, value: object) -> int:
        return int(self.address_type(value))

    def decode(self, bits: int) -> Address:
        return self.address_type(bits)

    def cidr(self, network: Network | str) -> BDDSet[Address]:
        block = self.network_type(network, strict=False)
        start = int(block.network_address)
        return BDDSet(
            self, IntSet.range(start, start + block.num_addresses, width=self.width)
        )

    def range(
        self, low: Address | str | int, high: Address | str | int
    ) -> BDDSet[Address]:
        """Addresses in `[low, high)`."""
        return BDDSet(
            self, IntSet.range(self.encode(low), self.encode(high), width=self.width)
        )

    def networks(self, source: BDDSet[Address]) -> Iterator[Network]:
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

        if source.codec != self:
            raise TypeError(f"{source.codec} addresses aren't {self} addresses")
        yield from walk(source.bits.bdd, self.width - 1, 0)


float16 = Float(16, "float16", exponent=5, format=">e")
float32 = Float(32, "float32", exponent=8, format=">f")
float64 = Float(64, "float64", exponent=11, format=">d")

ipv4 = IP(32, "ipv4", ipaddress.IPv4Address, ipaddress.IPv4Network)
ipv6 = IP(128, "ipv6", ipaddress.IPv6Address, ipaddress.IPv6Network)
