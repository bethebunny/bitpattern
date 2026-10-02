"""Implements `BDD`, a reduced, ordered binary decision diagram.

Diagrams and their algorithms are from Bryant's 1986 paper, [Graph-Based
Algorithms for Boolean Function Manipulation](https://doi.org/10.1109/TC.1986.1676819).
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from typing import ClassVar, TypeVar, TypeVarTuple, final

__all__ = ["BDD"]

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
InternKey = tuple[int, "BDD", "BDD"]


def weak_cache(fn: Callable[[*Ts], T]) -> Callable[[*Ts], T]:
    """Like functools.cache, but entries only live as long as their arguments."""
    table = weakref.WeakKeyDictionary()

    @functools.wraps(fn)
    def cached(*args: *Ts) -> T:
        *path, last = args
        entries = table
        for arg in path:
            if (inner := entries.get(arg)) is None:
                inner = entries[arg] = weakref.WeakKeyDictionary()
            entries = inner
        if (ref := entries.get(last)) is None or (result := ref()) is None:
            result = fn(*args)
            entries[last] = _reference(result)
        return result

    return cached


def _reference(value: T) -> Callable[[], T | None]:
    # Counts can't be weakly referenced, but they can't keep a node alive either.
    return weakref.ref(value) if isinstance(value, BDD) else lambda: value


@final
@dataclass(frozen=True, eq=False, repr=False, init=False)
class BDD:
    """A predicate on the bits of an integer as a reduced, ordered decision diagram.

    `BDD(bit, left, right)` is an expression on the `bit` boolean variable. It resolves
    to `left` for inputs where the bit is 0 and `right` where it is 1.

    All leaves in the tree must be `BDD.ACCEPT` or `BDD.REJECT`. BDDs are interned
    and guaranteed to produce the same object for the same expression.

    >>> odd = BDD(0, BDD.REJECT, BDD.ACCEPT)
    >>> ~odd is BDD(0, BDD.ACCEPT, BDD.REJECT)
    True
    >>> odd | ~odd
    BDD.ACCEPT

    Invariants:
        - **A node tests a higher bit than its children.**
            Every path tests bits in the same order, most significant first, so
            `&` and `|` can walk two diagrams together a bit at a time.
        - `BDD` instances are always reduced by construction.
        - `BDD` instances are interned. Constructing the same diagram will always
            yield exactly the same object.
    """

    bit: int
    left: BDD
    right: BDD

    ACCEPT: ClassVar[BDD]
    REJECT: ClassVar[BDD]
    intern: ClassVar[MutableMapping[InternKey, BDD]] = weakref.WeakValueDictionary()

    def __new__(cls, bit: int, left: BDD, right: BDD) -> BDD:
        if left is right:
            return left
        if left.bit >= bit or right.bit >= bit:
            raise ValueError(
                f"children must test bits below {bit}, not {left.bit} and {right.bit}"
            )
        key = (bit, left, right)
        if (node := cls.intern.get(key)) is None:
            node = cls.intern[key] = object.__new__(cls)
            vars(node).update(bit=bit, left=left, right=right)
        return node

    @property
    def leaf(self) -> bool:
        return self.bit == -1

    def __bool__(self) -> bool:
        return self is not BDD.REJECT

    @weak_cache
    def __invert__(self) -> BDD:
        if self.leaf:
            return BDD.REJECT if self else BDD.ACCEPT
        return BDD(self.bit, ~self.left, ~self.right)

    # operators are weakly cached, but need to be able to reject
    # inputs of invalid types which can't be weakly referenced
    def __and__(self, other: BDD) -> BDD:
        if not isinstance(other, BDD):
            return NotImplemented
        return self._and(other)

    def __or__(self, other: BDD) -> BDD:
        if not isinstance(other, BDD):
            return NotImplemented
        return self._or(other)

    def __sub__(self, other: BDD) -> BDD:
        if not isinstance(other, BDD):
            return NotImplemented
        return self & ~other

    @weak_cache
    def _and(self, other: BDD) -> BDD:
        if self is BDD.REJECT or other is BDD.ACCEPT:
            return self
        if other is BDD.REJECT or self is BDD.ACCEPT:
            return other
        bit = max(self.bit, other.bit)
        (a0, a1), (b0, b1) = cofactors(self, bit), cofactors(other, bit)
        return BDD(bit, a0 & b0, a1 & b1)

    @weak_cache
    def _or(self, other: BDD) -> BDD:
        if self is BDD.ACCEPT or other is BDD.REJECT:
            return self
        if other is BDD.ACCEPT or self is BDD.REJECT:
            return other
        bit = max(self.bit, other.bit)
        (a0, a1), (b0, b1) = cofactors(self, bit), cofactors(other, bit)
        return BDD(bit, a0 | b0, a1 | b1)

    # Leaves pickle by name, and nodes through the constructor so they get
    # re-interned.
    def __reduce__(self) -> str | tuple[type[BDD], tuple[int, BDD, BDD]]:
        if self.leaf:
            return repr(self)
        return BDD, (self.bit, self.left, self.right)

    def __repr__(self) -> str:
        if self.leaf:
            return "BDD.ACCEPT" if self else "BDD.REJECT"
        members = render_count(size(self))
        return f"<BDD bit={self.bit}, {node_count(self)} nodes, {members} members>"


def _leaf() -> BDD:
    leaf = object.__new__(BDD)
    vars(leaf).update(bit=-1, left=leaf, right=leaf)
    return leaf


BDD.ACCEPT, BDD.REJECT = _leaf(), _leaf()


def cofactors(node: BDD, bit: int) -> tuple[BDD, BDD]:
    """`node`'s branches for `bit` clear and set. `node` can't test above `bit`."""
    return (node.left, node.right) if node.bit == bit else (node, node)


@weak_cache
def size(node: BDD) -> int:
    """How many integers below `2 ** (node.bit + 1)` satisfy `node`."""
    if node.leaf:
        return int(node is BDD.ACCEPT)
    return count(node.left, node.bit) + count(node.right, node.bit)


def count(node: BDD, width: int) -> int:
    """How many integers below `2 ** width` satisfy `node`."""
    if node.bit >= width:
        raise ValueError(f"node tests bit {node.bit}, outside width {width}")
    return size(node) << (width - 1 - node.bit)  # each free bit above doubles it


def nth(node: BDD, width: int, n: int) -> int:
    """The `n`th smallest integer below `2 ** width` that satisfies `node`."""
    total = count(node, width)
    if not -total <= n < total:
        raise IndexError(n)
    return _nth(node, n % total)


def _nth(node: BDD, n: int) -> int:
    if node.leaf:
        return n  # every bit below is free, so the nth member is n
    # Free bits above the node are the high bits of the result, so they vary slowest.
    free, n = divmod(n, size(node))
    high = free << (node.bit + 1)
    if n < (left_count := count(node.left, node.bit)):
        return high | _nth(node.left, n)
    return high | 1 << node.bit | _nth(node.right, n - left_count)


def iterate(node: BDD, width: int) -> Iterator[int]:
    """Every integer below `2 ** width` that satisfies `node`, in order."""
    if node is BDD.ACCEPT:
        yield from range(1 << width)
    elif node:
        bit = width - 1
        low, high = cofactors(node, bit)
        yield from iterate(low, bit)
        for value in iterate(high, bit):
            yield 1 << bit | value


@weak_cache
def node_count(node: BDD) -> int:
    """How many nodes there are under `node`, not counting the leaves."""
    seen: set[BDD] = set()
    stack = [node]
    while stack:
        if not (current := stack.pop()).leaf and current not in seen:
            seen.add(current)
            stack += current.left, current.right
    return len(seen)


def render_count(total: int) -> str:
    """`total`, or `2**k` if it's a big power of two."""
    if total >= 1024 and not total & (total - 1):
        return f"2**{total.bit_length() - 1}"
    return str(total)


def pin(value: int, width: int) -> BDD:
    """Exactly `value`, over `width` bits."""
    node = BDD.ACCEPT
    for bit in range(width):
        if value >> bit & 1:
            node = BDD(bit, BDD.REJECT, node)
        else:
            node = BDD(bit, node, BDD.REJECT)
    return node


def less_than(bound: int, width: int) -> BDD:
    """Integers below `bound`, over `width` bits."""
    if bound <= 0:
        return BDD.REJECT
    if bound >= 1 << width:
        return BDD.ACCEPT
    node = BDD.REJECT
    for bit in range(width):
        # The highest bit where x and bound differ decides it. Where bound has a 1,
        # a 0 makes x smaller, and where bound has a 0, a 1 makes x bigger.
        if bound >> bit & 1:
            node = BDD(bit, BDD.ACCEPT, node)
        else:
            node = BDD(bit, node, BDD.REJECT)
    return node
