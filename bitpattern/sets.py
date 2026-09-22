"""`IntSet`: an immutable set of non-negative integers under a fixed bit width."""

from __future__ import annotations

import functools
import operator
import random
from collections.abc import Iterable, Iterator, Set as AbstractSet
from typing import TYPE_CHECKING, Self

from .bdd import BDD, BDDNode, REJECT, count, index, iterate, less_than, pin, render_count, zeros

if TYPE_CHECKING:
    from .pattern import Pattern

__all__ = ["IntSet"]

# A repr lists every member of a set this small, and otherwise just the ends.
REPR_MEMBERS, REPR_HEAD, REPR_TAIL = 10, 4, 2


class IntSet(AbstractSet):
    """An immutable set of non-negative integers under `2 ** width`, as a BDD.

    The diagram can't record the width -- free high bits have no node -- so it is
    carried alongside, and counting, iterating and indexing are relative to it.
    """

    __slots__ = ("bdd", "width")

    bdd: BDDNode
    width: int

    def __init__(self, iterable: Iterable[int] = (), width: int = 0):
        values = list(iterable)
        if values and (lowest := min(values)) < 0:
            raise ValueError(f"IntSet holds non-negative integers, not {lowest}")
        self.width = max([width, *(value.bit_length() for value in values)])
        self.bdd = functools.reduce(
            operator.or_, (pin(value, self.width) for value in values), REJECT
        )

    @classmethod
    def from_bdd(cls, bdd: BDDNode, width: int) -> Self:
        if bdd.bit >= width:
            raise ValueError(f"diagram tests bit {bdd.bit}, outside width {width}")
        self = cls.__new__(cls)
        self.bdd, self.width = bdd, width
        return self

    @classmethod
    def range(cls, start: int, stop: int, width: int | None = None) -> Self:
        """`{start, ..., stop - 1}`, in one node per bit however many it holds."""
        if start < 0:
            raise ValueError(f"IntSet holds non-negative integers, not {start}")
        if width is None:
            width = max(stop - 1, 0).bit_length()
        return cls.from_bdd(less_than(stop, width) - less_than(start, width), width)

    def _extended(self, width: int) -> BDDNode:
        # Members are bounded by `width`, but the diagram doesn't say so: read at
        # width 4, `{0}` from width 1 ("bit 0 clear") would be every even number.
        # So reading wider re-states the bound: the new high bits are clear.
        return self.bdd if width <= self.width else self.bdd & zeros(self.width, width)

    def _align(self, other: Iterable[int]) -> tuple[BDDNode, BDDNode, int]:
        # `IntSet`, not `type(self)`: a subclass constructor need not take an
        # iterable, and `Pattern`'s takes text.
        if not isinstance(other, IntSet):
            other = IntSet(other)
        width = max(self.width, other.width)
        return self._extended(width), other._extended(width), width

    def _canonical(self) -> tuple[BDDNode, int]:
        """The same members at their minimal width, so equal sets share a key."""
        width = self[-1].bit_length() if self else 0
        bdd = self.bdd
        while bdd.bit >= width:
            bdd = bdd.left  # every member fits in `width`, so `right` is empty
        return bdd, width

    def widen(self, width: int) -> Self:
        """The same members, over at least `width` bits."""
        if width <= self.width:
            return self
        return type(self).from_bdd(self._extended(width), width)

    @property
    def pattern(self) -> Pattern:
        from .pattern import Pattern

        return Pattern.from_bdd(self.bdd, self.width)

    @property
    def size(self) -> int:
        """How many members. Unlike `len`, not capped at `2 ** 63 - 1`."""
        return count(self.bdd, self.width - 1)

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        # Otherwise truth-testing falls back to `__len__`, which can overflow.
        return self.bdd is not REJECT

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int) or value < 0 or value.bit_length() > self.width:
            return False
        node = self.bdd
        while isinstance(node, BDD):
            node = node.right if value >> node.bit & 1 else node.left
        return bool(node)

    def __iter__(self) -> Iterator[int]:
        return iterate(self.bdd, self.width - 1)

    def __getitem__(self, item: int) -> int:
        return index(self.bdd, self.width - 1, item)

    def sample(self, rng: random.Random | None = None) -> int:
        """A uniformly drawn member, without enumerating."""
        if not self:
            raise KeyError("sample from an empty IntSet")
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

    # The `Set` mixins' reflected operators iterate both operands, so `{1} | huge`
    # would never return.
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    def __rsub__(self, other: Iterable[int]) -> IntSet:
        if not isinstance(other, Iterable):
            return NotImplemented
        return IntSet(other) - self

    def __invert__(self) -> Self:
        """Every non-member under `2 ** width`."""
        return type(self).from_bdd(~self.bdd, self.width)

    # Comparisons take only `IntSet`s: equality must, since a hash consistent with
    # frozenset's would be O(members), and the orderings follow so that
    # `a <= b <= a` still means `a == b`.
    def __le__(self, other: object) -> bool:
        if not isinstance(other, IntSet): return NotImplemented
        return not self - other

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, IntSet): return NotImplemented
        return not other - self

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, IntSet): return NotImplemented
        return self <= other and self != other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, IntSet): return NotImplemented
        return self >= other and self != other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntSet): return NotImplemented
        return self._canonical() == other._canonical()

    def __hash__(self) -> int:
        return hash(self._canonical())

    def isdisjoint(self, other: Iterable[int]) -> bool:
        return not self & other

    def __repr__(self) -> str:
        total = self.size
        if total <= REPR_MEMBERS:
            return f"IntSet({list(self)!r}, width={self.width})"
        head = ", ".join(repr(self[at]) for at in range(REPR_HEAD))
        tail = ", ".join(repr(self[at]) for at in range(-REPR_TAIL, 0))
        return f"IntSet([{head}, ..., {tail}], size={render_count(total)}, width={self.width})"
