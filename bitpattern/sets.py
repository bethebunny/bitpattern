"""IntSet, an immutable set of non-negative integers backed by a BDD."""

from __future__ import annotations

import functools
import itertools
import operator
import random
from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Self, overload

from .bdd import BDD, count, iterate, less_than, nth, pin, render_count

if TYPE_CHECKING:
    from .pattern import Pattern

__all__ = ["IntSet"]


class IntSet(AbstractSet[int], Sequence[int]):
    """An immutable set of non-negative integers below `2 ** width`.

    It's also a sequence of its members in order, so it can be indexed and sliced.
    Free high bits don't have nodes, so a diagram can't say how wide its set is.
    The set keeps track of that instead.
    """

    __slots__ = ("bdd", "width")

    bdd: BDD
    width: int

    def __init__(self, iterable: Iterable[int] = (), width: int = 0) -> None:
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
    def range(cls, start: int, stop: int, width: int | None = None) -> Self:
        """The integers in `range(start, stop)`, in one node per bit."""
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
        # Not type(self)(other), since Pattern's constructor takes text.
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
    def pattern(self) -> Pattern:
        from .pattern import Pattern  # circular: Pattern is an IntSet

        return Pattern.from_bdd(self.bdd, self.width)

    @property
    def size(self) -> int:
        """How many members there are. Unlike len(), this can go past `2**63 - 1`."""
        return count(self.bdd, self.width)

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        # Otherwise bool() uses len(), which can overflow.
        return bool(self.bdd)

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int) or value < 0 or value.bit_length() > self.width:
            return False
        node = self.bdd
        while node.bit >= 0:
            node = node.right if value >> node.bit & 1 else node.left
        return bool(node)

    def __iter__(self) -> Iterator[int]:
        return iterate(self.bdd, self.width)

    def __reversed__(self) -> Iterator[int]:
        # Sequence's goes through len(), which can overflow.
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

    def choice(self, rng: random.Random | None = None) -> int:
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

    # Only compare with other IntSets. Being equal to a frozenset would mean hashing
    # like one, which is O(members). The orderings follow suit, so that a <= b <= a
    # still means a == b.
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
