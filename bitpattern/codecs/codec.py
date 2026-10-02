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
        """The empty set."""
        return BDDSet(self, IntSet.from_bdd(BDD.REJECT, self.width))

    def set(self, values: Iterable[T] = ()) -> BDDSet[T]:
        """Constructs a BDDSet from the input values."""
        return BDDSet(self, IntSet(map(self.encode, values), width=self.width))

    def pattern(self, text: str) -> BDDSet[T]:
        """Constructs a BDDSet of values matching the pattern."""
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
    storage: IntSet

    def __post_init__(self) -> None:
        if self.storage.width != self.codec.width:
            raise ValueError(
                f"{self.codec} needs {self.codec.width} bits, not {self.storage.width}"
            )

    def _storage(self, other: object) -> IntSet:
        if not isinstance(other, BDDSet) or other.codec != self.codec:
            # Raise rather than return NotImplemented, since IntSet's reflected
            # operators would try to enumerate the set.
            raise TypeError(f"{self.codec} sets only combine with each other")
        return other.storage

    @property
    def size(self) -> int:
        return self.storage.size

    def __len__(self) -> int:
        return len(self.storage)

    def __bool__(self) -> bool:
        return bool(self.storage)

    def __contains__(self, value: object) -> bool:
        try:
            return self.codec.encode(value) in self.storage
        except ValueError:  # not something this codec can encode
            return False

    def __iter__(self) -> Iterator[T]:
        return map(self.codec.decode, self.storage)

    def __reversed__(self) -> Iterator[T]:
        return map(self.codec.decode, reversed(self.storage))

    @overload
    def __getitem__(self, item: int) -> T: ...
    @overload
    def __getitem__(self, item: slice) -> BDDSet[T]: ...
    def __getitem__(self, item: int | slice) -> T | BDDSet[T]:
        if isinstance(item, slice):
            return BDDSet(self.codec, self.storage[item])
        return self.codec.decode(self.storage[item])

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        return self.storage.index(self.codec.encode(value), start, stop)

    def count(self, value: object) -> int:
        return int(value in self)

    def choice(self, *, rng: random.Random | None = None) -> T:
        """random.choice(self), but supports sets larger than 2**63."""
        return self.codec.decode(self.storage.choice(rng=rng))

    def __and__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self.storage & self._storage(other))

    def __or__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self.storage | self._storage(other))

    def __sub__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self.storage - self._storage(other))

    def __xor__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self.storage ^ self._storage(other))

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    def __rsub__(self, other: AbstractSet[T]) -> BDDSet[T]:
        return BDDSet(self.codec, self._storage(other) - self.storage)

    def __invert__(self) -> BDDSet[T]:
        """Every other bit pattern this codec has."""
        return BDDSet(self.codec, ~self.storage)

    def __le__(self, other: object) -> bool:
        return self.storage <= self._storage(other)

    def __lt__(self, other: object) -> bool:
        return self.storage < self._storage(other)

    def __ge__(self, other: object) -> bool:
        return self.storage >= self._storage(other)

    def __gt__(self, other: object) -> bool:
        return self.storage > self._storage(other)

    def isdisjoint(self, other: Iterable[T]) -> bool:
        return self.storage.isdisjoint(self._storage(other))

    def __repr__(self) -> str:
        # Render small sets directly. Larger sets print a summary.
        if (size := self.size) <= 10:
            return f"BDDSet({self.codec}, {list(self)})"
        head = ", ".join(map(repr, itertools.islice(self, 4)))
        return (
            f"BDDSet({self.codec}, [{head}, ..., {self[-2]!r}, {self[-1]!r}], "
            f"size={render_count(size)})"
        )
