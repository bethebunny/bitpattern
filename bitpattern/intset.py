"""IntSet, an immutable set of non-negative integers backed by a BDD."""

from __future__ import annotations

import functools
import itertools
import operator
import random
from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from typing import Self, overload

from .bdd import BDD, count, iterate, less_than, nth, pin, render_count

__all__ = ["IntSet"]


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

    Most operations are O(w) in the width, however many members there are. `&`,
    `|`, `-` and `^` are O(|a| * |b|) in the number of nodes in each diagram, and
    `~` is O(|a|). A diagram has at most `size * w` nodes, and usually far fewer,
    eg. any `Pattern(text)` has at most one node per bit.

    An IntSet is a diagram and a width. Its members are the integers below
    `2 ** width` whose bits satisfy the diagram. Free bits don't need nodes, so the
    diagram doesn't know the width. For instance `IntSet([0], width=1)` is "bit 0
    is clear", which read at width 4 would be every even number. Sets of different
    widths are widened to match before they're combined, which adds that bound to
    the diagram explicitly.
    """

    __slots__ = ("bdd", "width")

    bdd: BDD
    width: int

    def __init__(self, iterable: Iterable[int] = (), *, width: int = 0) -> None:
        values = list(iterable)
        if values and (lowest := min(values)) < 0:
            raise ValueError(f"IntSets can't hold negative numbers, got {lowest}")
        self.width = max(width, max(values, default=0).bit_length())
        pins = (pin(value, self.width) for value in values)
        self.bdd = functools.reduce(operator.or_, pins, BDD.REJECT)

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
            raise ValueError(f"IntSets can't hold negative numbers, got {start}")
        if width is None:
            width = max(stop - 1, 0).bit_length()
        return cls.from_bdd(less_than(stop, width) - less_than(start, width), width)

    def _extended(self, width: int) -> BDD:
        # The width bounds the members, but the diagram doesn't know that: {0} at
        # width 1 is "bit 0 is clear", which read at width 4 is every even number.
        # So reading it wider needs the bound added explicitly.
        if width <= self.width:
            return self.bdd
        return self.bdd & less_than(1 << self.width, width)

    def _align(self, other: Iterable[int]) -> tuple[BDD, BDD, int]:
        if not isinstance(other, IntSet):
            other = IntSet(other)
        width = max(self.width, other.width)
        return self._extended(width), other._extended(width), width

    def _canonical(self) -> tuple[BDD, int]:
        """The diagram at the narrowest width that fits, so equal sets match."""
        width = self[-1].bit_length() if self else 0
        bdd = self.bdd
        while bdd.bit >= width:
            bdd = bdd.left  # all the members fit in width, so the right branch is empty
        return bdd, width

    def widen(self, width: int) -> Self:
        """The same members, over at least `width` bits."""
        if width <= self.width:
            return self
        return type(self).from_bdd(self._extended(width), width)

    @property
    def size(self) -> int:
        """How many members there are. Unlike len(), this can go past `2**63 - 1`."""
        return count(self.bdd, self.width)

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return bool(self.bdd)

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int) or value < 0 or value.bit_length() > self.width:
            return False
        node = self.bdd  # XXX: if things like `iterate` are free functions in BDD, contains probably can be too.
        while node.bit >= 0:
            node = node.right if value >> node.bit & 1 else node.left
        return bool(node)

    def __iter__(self) -> Iterator[int]:
        return iterate(self.bdd, self.width)

    def __reversed__(self) -> Iterator[int]:
        # Sequence's goes through len() which can overflow.
        return map(self.__getitem__, reversed(range(self.size)))

    @overload
    def __getitem__(self, item: int) -> int: ...
    @overload
    def __getitem__(self, item: slice) -> Self: ...
    def __getitem__(self, item: int | slice) -> int | Self:
        if not isinstance(item, slice):
            return nth(self.bdd, self.width, item)
        # A slice is the members at those positions. They're still a set, so a
        # negative step doesn't reverse anything.
        positions = range(self.size)[item]
        if positions.step == 1 and positions:  # one range, so no need to find each one
            return self & IntSet.range(self[positions[0]], self[positions[-1]] + 1)
        return self & IntSet(map(self.__getitem__, positions))

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        """Where `value` is among the members, without searching for it."""
        if isinstance(value, int) and value in self:
            # Its position is how many members are below it.
            at = count(self.bdd & less_than(value, self.width), self.width)
            if at in range(self.size)[start:stop]:
                return at
        raise ValueError(f"{value!r} is not in the set")

    def count(self, value: object) -> int:
        return int(value in self)

    def choice(self, *, rng: random.Random | None = None) -> int:
        """A uniformly random member, like random.choice() but at any size."""
        if not self:
            raise IndexError("can't choose from an empty set")
        return self[(rng or random).randrange(self.size)]

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
        return type(self).from_bdd((a | b) - (a & b), width)

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
        return not self - other

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, IntSet):
            return NotImplemented
        return not other - self

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
        return not self & other

    def __repr__(self) -> str:
        # Small sets repr every member, and bigger ones just a few from each end.
        if (size := self.size) <= 10:
            return f"IntSet({list(self)}, width={self.width})"
        head = ", ".join(map(str, itertools.islice(self, 4)))
        return (
            f"IntSet([{head}, ..., {self[-2]}, {self[-1]}], "
            f"size={render_count(size)}, width={self.width})"
        )
