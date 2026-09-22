"""`IntSet`: an immutable set of non-negative integers under a fixed bit width."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Set as AbstractSet
from typing import TYPE_CHECKING, Self

from .bdd import (
    BDD,
    BDDNode,
    REJECT,
    count,
    index,
    iterate,
    less_than,
    pin,
    render_count,
    zeros,
)

if TYPE_CHECKING:
    from .pattern import Pattern

__all__ = ["IntSet"]

# How many members a repr will spell out in full before it starts truncating,
# and how many to show at each end once it does.
EXACT, HEAD, TAIL = 10, 4, 2


class IntSet(AbstractSet):
    """A set of non-negative integers under `2 ** width`, stored as a BDD.

    The diagram alone cannot say how wide the universe is -- a pattern whose top
    bits are free has no node for them -- so the width is carried alongside it
    and everything that counts, enumerates or indexes is relative to it.

    Members are bounded by the width, but nothing in the predicate records that
    bound. Widening therefore has to re-state it, by conjoining "the new high
    bits are clear"; otherwise `{0}` at width 1 would silently become every even
    number below 16 when read at width 4.

    Instances are immutable, which is what lets them be hashed and cached: the
    diagram underneath is interned, so hashing is O(1) however many members the
    set has. Rebind rather than mutate -- `group |= {9}` works and is cheap,
    because the old diagram is shared rather than copied.
    """

    bdd: BDDNode
    width: int

    __slots__ = ("bdd", "width", "_form")

    def __init__(self, iterable: Iterable[int] = (), width: int = 0):
        bdd: BDDNode = REJECT
        for value in iterable:
            if value < 0:
                raise ValueError(f"IntSet holds non-negative integers, not {value}")
            if value.bit_length() > width:
                bdd = bdd & zeros(width, value.bit_length())
                width = value.bit_length()
            bdd = bdd | pin(value, width)
        self.bdd = bdd
        self.width = width
        self._form = None

    @classmethod
    def from_bdd(cls, bdd: BDDNode, width: int) -> Self:
        """Wrap a predicate directly. The escape hatch `codecs` builds through."""
        if bdd.bit >= width:
            raise ValueError(f"diagram tests bit {bdd.bit}, outside width {width}")
        self = cls.__new__(cls)
        self.bdd = bdd
        self.width = width
        self._form = None
        return self

    @classmethod
    def range(cls, start: int, stop: int, width: int | None = None) -> IntSet:
        """`{start, ..., stop - 1}`, half-open like the builtin `range`.

        Built as a difference of two comparisons rather than element by element,
        so it costs one node per bit however many integers it holds.
        """
        if start < 0:
            raise ValueError(f"IntSet holds non-negative integers, not {start}")
        if width is None:
            width = max(stop - 1, 0).bit_length()
        stop = max(start, stop)
        return cls.from_bdd(less_than(stop, width) - less_than(start, width), width)

    @classmethod
    def _coerce(cls, other: Iterable[int]) -> IntSet:
        # Deliberately `IntSet`, not `cls`: a subclass constructor need not take
        # an iterable -- `Pattern` takes text -- and `cls(other)` would then
        # build the wrong set silently instead of failing.
        return other if isinstance(other, IntSet) else IntSet(other)

    @classmethod
    def _from_iterable(cls, iterable: Iterable[int]) -> IntSet:
        """What the `Set` mixins build results with. Same reasoning as `_coerce`."""
        return IntSet(iterable)

    def _extended(self, width: int) -> BDDNode:
        """This set's predicate, restated over a universe of `width` bits."""
        return self.bdd if width <= self.width else self.bdd & zeros(self.width, width)

    def _align(self, other: Iterable[int]) -> tuple[BDDNode, BDDNode, int]:
        if not isinstance(other, Iterable):
            raise TypeError(f"expected an iterable of integers, not {type(other).__name__}")
        other = self._coerce(other)
        width = max(self.width, other.width)
        return self._extended(width), other._extended(width), width

    def _canonical(self) -> tuple[BDDNode, int]:
        """The same members at their minimal width, so equal sets share a key.

        Equality is by members, so a set is equal to the same members held at a
        wider universe -- but those are different diagrams, and hashing them
        apart would break the invariant. Stripping the forced-clear high bits
        gives one representative per member set.
        """
        if self._form is None:
            width = self[-1].bit_length() if self.size else 0
            bdd = self.bdd
            while bdd.bit >= width:
                # Every member fits in `width` bits, so this one is forced clear
                # and the whole `right` branch is unsatisfiable.
                bdd = bdd.left
            self._form = (bdd, width)
        return self._form

    def widen(self, width: int) -> IntSet:
        """The same members over a universe of at least `width` bits."""
        if width <= self.width:
            return self
        return type(self).from_bdd(self._extended(width), width)

    @property
    def pattern(self) -> Pattern:
        """This set written as a union of bit patterns."""
        from .pattern import Pattern

        return Pattern.from_bdd(self.bdd, self.width)

    @property
    def size(self) -> int:
        """How many members, as an unbounded int.

        This, not `len`, is the primitive: `__len__` must fit a `Py_ssize_t`, and
        a 64-bit universe holds more members than that.
        """
        return count(self.bdd, self.width - 1)

    def __len__(self) -> int:
        """`size`, narrowed to what the sequence protocol can carry.

        Raises `OverflowError` past `2 ** 63 - 1`; use `size` for those.
        """
        return self.size

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int) or value < 0 or value.bit_length() > self.width:
            return False
        node = self.bdd
        while isinstance(node, BDD):
            node = node.right if (value >> node.bit) & 1 else node.left
        return bool(node)

    def __iter__(self) -> Iterator[int]:
        return iterate(self.bdd, self.width - 1)

    def __getitem__(self, item: int) -> int:
        return index(self.bdd, self.width - 1, item)

    def sample(self, rng=random) -> int:
        """A member drawn uniformly, in O(width) and without enumerating."""
        if not (total := self.size):
            raise KeyError("sample from an empty IntSet")
        return self[rng.randrange(total)]

    def __and__(self, other: Iterable[int]) -> IntSet:
        a, b, width = self._align(other)
        return type(self).from_bdd(a & b, width)

    def __or__(self, other: Iterable[int]) -> IntSet:
        a, b, width = self._align(other)
        return type(self).from_bdd(a | b, width)

    def __sub__(self, other: Iterable[int]) -> IntSet:
        a, b, width = self._align(other)
        return type(self).from_bdd(a - b, width)

    def __invert__(self) -> IntSet:
        """Every non-member under `2 ** width`."""
        return type(self).from_bdd(~self.bdd, self.width)

    # The `Set` mixins answer these by iterating and by calling `len`, which is
    # O(members) and overflows on a wide universe. The diagram answers in
    # O(nodes): one set is inside another exactly when the difference is empty.
    def __le__(self, other: Iterable[int]) -> bool:
        a, b, _ = self._align(other)
        return a - b is REJECT

    def __ge__(self, other: Iterable[int]) -> bool:
        a, b, _ = self._align(other)
        return b - a is REJECT

    def __lt__(self, other: Iterable[int]) -> bool:
        a, b, _ = self._align(other)
        return a is not b and a - b is REJECT

    def __gt__(self, other: Iterable[int]) -> bool:
        a, b, _ = self._align(other)
        return a is not b and b - a is REJECT

    def isdisjoint(self, other: Iterable[int]) -> bool:
        a, b, _ = self._align(other)
        return a & b is REJECT

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntSet):
            return self._canonical() == other._canonical()
        if isinstance(other, AbstractSet):
            return self._canonical() == IntSet(other)._canonical()
        return NotImplemented

    def __hash__(self) -> int:
        # Not `Set._hash`, which is O(members) and would never return on a set of
        # 2**53 floats. Interning makes the canonical diagram hashable in O(1).
        return hash(self._canonical())

    def __repr__(self) -> str:
        """Bounded, and never enumerating -- indexing is O(width), so the head
        and tail of an arbitrarily large set come for free."""
        total = self.size
        if total <= EXACT:
            return f"IntSet({list(self)!r}, width={self.width})"
        head = ", ".join(repr(self[at]) for at in range(HEAD))
        tail = ", ".join(repr(self[at]) for at in range(-TAIL, 0))
        return (
            f"IntSet([{head}, ..., {tail}], "
            f"size={render_count(total)}, width={self.width})"
        )
