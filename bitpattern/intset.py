"""IntSet, an immutable set of non-negative integers backed by a BDD."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Self, overload

from .bdd import (
    BDD,
    accepts,
    count,
    count_repr,
    disjoint,
    exactly_these,
    interval,
    members,
    narrowed,
    nth,
    rank,
    subset,
)
from .dlpack import CPU, exported, unsigned

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

__all__ = ["IntSet"]


def written(value: object) -> str:
    """`repr(value)`, with ints too long for decimal in hex (alone or in a plain
    tuple)."""
    try:
        return repr(value)
    except ValueError:
        if isinstance(value, int):
            return hex(value)
        if type(value) is tuple:
            items = ", ".join(map(written, value))
            return f"({items},)" if len(value) == 1 else f"({items})"
        raise


def described(value: int) -> str:
    """`str(value)`, or its size if it's too long for `str`."""
    try:
        return str(value)
    except ValueError:
        return f"an int of {value.bit_length()} bits"


def listed(values: Iterable[object]) -> str:
    values = list(values)
    try:
        return repr(values)[1:-1]  # quicker than writing each one out
    except ValueError:
        return ", ".join(map(written, values))


class IntSet(AbstractSet[int], Sequence[int]):
    """A sequential, immutable set of integers in `[0, 2 ** width)`.

    IntSet is backed by BDDs and can efficiently operate on very large sets. It's a
    `collections.abc.Set`, and also a `Sequence` of its members in sorted order.
    Slicing gives back a set.

    >>> s = IntSet.range(3, 17)
    >>> s
    IntSet([3, 4, 5, 6, ..., 15, 16], size=14, width=5)
    >>> s[2]
    5
    >>> s[2:5]
    IntSet([5, 6, 7], width=5)
    >>> s & IntSet([1, 2, 3, 4])
    IntSet([3, 4], width=5)
    >>> IntSet.range(0, 2**100, width=128).size
    1267650600228229401496703205376

    Python isn't really designed for data structures larger than memory. In
    particular many operations will fail if `__len__` returns a number >= 2**63.
    When working with very large sets:

    - Use `s.size` instead of `len(s)`
    - Use `s.choice()` instead of `random.choice(s)` or `random.sample(s, k)`
    - Use `s[0]` and `s[-1]` instead of `min(s)` and `max(s)`, which look at
      every member
    - Use `bitpattern.strategies` instead of hypothesis's `sampled_from(s)`

    For width w and diagrams of |a| and |b| nodes:

    - Most operations are O(w), however many members there are
    - `&`, `|`, `-` and `^` are O(|a| * |b|)
    - `~` is O(|a|)

    A diagram has at most `size * w` nodes and usually far fewer, eg. a pattern has
    at most one node per bit.

    An IntSet is a diagram and a width, and holds the integers below `2 ** width`
    whose bits satisfy the diagram. Sets of different widths are widened to match
    before being combined.
    """

    __slots__ = ("bdd", "width")

    bdd: BDD
    width: int

    def __init__(self, iterable: Iterable[int] = (), *, width: int = 0) -> None:
        self.bdd, self.width = exactly_these(iterable, width=width)

    @classmethod
    def from_bdd(cls, bdd: BDD, width: int) -> Self:
        if bdd.bit >= width:
            raise ValueError(f"diagram tests bit {bdd.bit}, outside width {width}")
        self = cls.__new__(cls)
        self.bdd, self.width = bdd, width
        return self

    @classmethod
    def range(cls, start: int, stop: int, *, width: int | None = None) -> Self:
        if start < 0:
            raise ValueError(
                f"IntSets can't hold negative numbers, got {described(start)}"
            )
        if width is None:
            width = max(stop - 1, 0).bit_length()
        return cls.from_bdd(interval(start, stop, width=width), width)

    def _extended(self, width: int) -> BDD:
        # The diagram doesn't encode the width, eg. {0} at width 1 is "bit 0 is clear",
        # which at width 4 is every even number.
        if width <= self.width:
            return self.bdd
        return self.bdd & interval(0, 1 << self.width, width=width)

    def _align(self, other: Iterable[int]) -> tuple[BDD, BDD, int]:
        if not isinstance(other, IntSet):
            other = IntSet(other)
        width = max(self.width, other.width)
        return self._extended(width), other._extended(width), width

    def _canonical(self) -> tuple[BDD, int]:
        """The diagram at the narrowest width that fits, so equal sets match."""
        width = self[-1].bit_length() if self else 0
        return narrowed(self.bdd, width=width), width

    def widened(self, width: int) -> Self:
        """The same members, over at least `width` bits."""
        if width <= self.width:
            return self
        return type(self).from_bdd(self._extended(width), width)

    @property
    def size(self) -> int:
        """How many members there are. Unlike len(), this can go past `2**63 - 1`."""
        return count(self.bdd, width=self.width)

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return bool(self.bdd)

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int) or value < 0 or value.bit_length() > self.width:
            return False
        return accepts(self.bdd, value)

    def __iter__(self) -> Iterator[int]:
        return members(self.bdd, width=self.width)

    def __reversed__(self) -> Iterator[int]:
        # Sequence's goes through len() which can overflow.
        return map(self.__getitem__, reversed(range(self.size)))

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType:
        """The members as a DLPack tensor of the narrowest unsigned int that fits. Past
        64 bits each member is a row of bytes, most significant first."""
        return exported(
            self.bdd,
            width=self.width,
            dtype=unsigned(self.width),
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        return (CPU, 0)

    @overload
    def __getitem__(self, item: int) -> int: ...
    @overload
    def __getitem__(self, item: slice) -> Self: ...
    def __getitem__(self, item: int | slice) -> int | Self:
        if not isinstance(item, slice):
            return nth(self.bdd, item, width=self.width)
        # A set, so a negative step doesn't reverse anything.
        positions = range(self.size)[item]
        if positions.step == 1 and positions:
            return self & IntSet.range(self[positions[0]], self[positions[-1]] + 1)
        return self & IntSet(map(self.__getitem__, positions))

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        """O(w), without searching."""
        if isinstance(value, int) and value in self:
            at = rank(self.bdd, value, width=self.width)
            if at in range(self.size)[start:stop]:
                return at
        raise ValueError(f"{value!r} is not in the set")

    def count(self, value: object) -> int:
        return int(value in self)

    def choice(self, *, rng: random.Random | None = None) -> int:
        """Like `random.choice`, but at any size."""
        if not (size := self.size):
            raise IndexError("can't choose from an empty set")
        return nth(self.bdd, (rng or random).randrange(size), width=self.width)

    def __and__(self, other: Iterable[int]) -> Self:
        a, b, width = self._align(other)
        return type(self).from_bdd(a & b, width)

    def __or__(self, other: Iterable[int]) -> Self:
        a, b, width = self._align(other)
        return type(self).from_bdd(a | b, width)

    def __sub__(self, other: Iterable[int]) -> Self:
        a, b, width = self._align(other)
        return type(self).from_bdd(a - b, width)

    def __xor__(self, other: Iterable[int]) -> Self:
        a, b, width = self._align(other)
        return type(self).from_bdd(a ^ b, width)

    # Set's own reflected operators iterate both sides, so {1} | huge would hang.
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    def __rsub__(self, other: Iterable[int]) -> IntSet:
        return IntSet(other) - self

    def __invert__(self) -> Self:
        """Everything below `2 ** width` that isn't a member."""
        return type(self).from_bdd(~self.bdd, self.width)

    # Only compare with other IntSets. Being equal to eg. a frozenset would mean
    # hashing like one, which is O(members).
    def __le__(self, other: object) -> bool:
        if not isinstance(other, IntSet):
            return NotImplemented
        a, b, _ = self._align(other)
        return subset(a, b)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, IntSet):
            return NotImplemented
        a, b, _ = self._align(other)
        return subset(b, a)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, IntSet):
            return NotImplemented
        return self <= other and self != other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, IntSet):
            return NotImplemented
        return self >= other and self != other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntSet):
            return NotImplemented
        if self.width == other.width:
            return self.bdd is other.bdd
        return self._canonical() == other._canonical()

    def __hash__(self) -> int:
        return hash(self._canonical())

    def isdisjoint(self, other: Iterable[int]) -> bool:
        a, b, _ = self._align(other)
        return disjoint(a, b)

    def __repr__(self) -> str:
        if (size := self.size) <= 10:
            return f"IntSet([{listed(self)}], width={self.width})"
        head, tail = listed(itertools.islice(self, 4)), listed([self[-2], self[-1]])
        return (
            f"IntSet([{head}, ..., {tail}], "
            f"size={count_repr(size)}, width={self.width})"
        )
