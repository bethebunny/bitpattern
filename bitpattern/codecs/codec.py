"""Codec, an encoding of values as integers, and BDDSet, the sets of values it makes."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar, overload

from ..bdd import BDD, render_count
from ..intset import IntSet
from ..pattern import Pattern

__all__ = ["BDDSet", "Codec"]

T = TypeVar("T")


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
