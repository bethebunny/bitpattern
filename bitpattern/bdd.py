from __future__ import annotations

import enum
import functools
import operator
import weakref
from dataclasses import dataclass
from typing import Callable, ClassVar, Iterator

__all__ = ["ACCEPT", "BDD", "BDDLeaf", "BDDNode", "REJECT"]


def weak_cache(fn):
    """Memoise `fn` for exactly as long as its arguments and result are alive.

    BDD `apply` is exponential without memoisation, but a cache holding nodes
    strongly would keep every one it had seen alive and defeat the weak
    interning. Results are held weakly too: `a & ACCEPT` is `a`, and a strong
    result would keep its own key alive.
    """
    table = weakref.WeakKeyDictionary()

    @functools.wraps(fn)
    def cached(*args):
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


def _reference(value):
    # Nodes are held weakly; counts are ints, which can't be, and hold nothing.
    return weakref.ref(value) if isinstance(value, BDDNode) else lambda: value


class BDDNode:
    """A reduced, ordered binary decision diagram: a predicate over integer bits.

    `bit` is the bit tested, `left` the branch taken when it is clear and `right`
    when it is set. Bits no node tests are free. Nodes are interned, so equal
    diagrams are the same object.
    """

    bit: int

    def __invert__(self) -> BDDNode: raise NotImplementedError
    def __and__(self, other: BDDNode) -> BDDNode: raise NotImplementedError
    def __or__(self, other: BDDNode) -> BDDNode: raise NotImplementedError

    def __sub__(self, other: BDDNode) -> BDDNode:
        if not isinstance(other, BDDNode): return NotImplemented
        return self & ~other


class BDDLeaf(BDDNode, enum.Enum):
    REJECT = False
    ACCEPT = True

    @property
    def bit(self) -> int:
        return -1

    def __bool__(self):
        return self.value

    def __invert__(self):
        return BDDLeaf(not self.value)

    def __and__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        return self and other

    def __or__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        return self or other

    def __repr__(self):
        return self.name

    __str__ = __repr__


ACCEPT, REJECT = BDDLeaf.ACCEPT, BDDLeaf.REJECT


@dataclass(frozen=True, eq=False, repr=False)
class BDD(BDDNode):
    bit: int
    left: BDDNode
    right: BDDNode

    intern: ClassVar[weakref.WeakValueDictionary[tuple, BDD]] = weakref.WeakValueDictionary()

    def __new__(cls, bit: int, left: BDDNode, right: BDDNode) -> BDDNode:
        if left is right: return left
        if left.bit >= bit or right.bit >= bit:
            raise ValueError(f"children of bit {bit} must test lower bits, not {left.bit} and {right.bit}")
        key = (bit, left, right)
        if (node := cls.intern.get(key)) is None:
            node = cls.intern[key] = object.__new__(cls)
            object.__setattr__(node, "bit", bit)
            object.__setattr__(node, "left", left)
            object.__setattr__(node, "right", right)
        return node

    # Nodes are built once, in `__new__`. The dataclass `__init__` would rewrite a
    # shared node's fields -- including the child that `left is right` returns.
    def __init__(self, *_): pass

    # Rebuild through the constructor, so unpickling re-interns.
    def __reduce__(self):
        return BDD, (self.bit, self.left, self.right)

    def __invert__(self):
        return _negate(self)

    def __and__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        return _and(self, other)

    def __or__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        return _or(self, other)

    def __repr__(self):
        return f"<BDD bit={self.bit}, {node_count(self)} nodes, {render_count(size(self))} members>"


@weak_cache
def _negate(node: BDD) -> BDDNode:
    return BDD(node.bit, ~node.left, ~node.right)


@weak_cache
def _and(a: BDD, b: BDDNode) -> BDDNode:
    return _apply(operator.and_, a, b)


@weak_cache
def _or(a: BDD, b: BDDNode) -> BDDNode:
    return _apply(operator.or_, a, b)


def _apply(op: Callable[[BDDNode, BDDNode], BDDNode], a: BDD, b: BDDNode) -> BDDNode:
    # The higher bit goes outermost, and a leaf settles the result on its own.
    if isinstance(b, BDDLeaf) or a.bit < b.bit:
        return op(b, a)
    if a.bit > b.bit:
        return BDD(a.bit, op(a.left, b), op(a.right, b))
    return BDD(a.bit, op(a.left, b.left), op(a.right, b.right))


def cofactors(node: BDDNode, bit: int) -> tuple[BDDNode, BDDNode]:
    """`node` given `bit` clear, and given it set. `node` may not test above `bit`."""
    return (node, node) if node.bit < bit else (node.left, node.right)


@weak_cache
def size(node: BDDNode) -> int:
    """How many integers in `[0, 2 ** (node.bit + 1))` satisfy `node`."""
    if isinstance(node, BDDLeaf):
        return int(node.value)
    return count(node.left, node.bit - 1) + count(node.right, node.bit - 1)


def count(node: BDDNode, top: int) -> int:
    """How many integers in `[0, 2 ** (top + 1))` satisfy `node`."""
    if node.bit > top:
        raise ValueError(f"node tests bit {node.bit}, above top bit {top}")
    return size(node) << (top - node.bit)  # each free bit above doubles it


def index(node: BDDNode, top: int, item: int) -> int:
    """The `item`-th smallest integer in `[0, 2 ** (top + 1))` satisfying `node`."""
    total = count(node, top)
    if not -total <= item < total:
        raise IndexError(item)
    return _index(node, item % total)


def _index(node: BDDNode, item: int) -> int:
    if isinstance(node, BDDLeaf):
        return item  # ACCEPT: the n-th member of its span is n
    # Free bits above `node.bit` are the high bits of the result, so they vary slowest.
    free, item = divmod(item, size(node))
    high = free << (node.bit + 1)
    if item < (low := count(node.left, node.bit - 1)):
        return high | _index(node.left, item)
    return high | 1 << node.bit | _index(node.right, item - low)


def iterate(node: BDDNode, top: int) -> Iterator[int]:
    """Every integer in `[0, 2 ** (top + 1))` satisfying `node`, ascending."""
    if node is REJECT:
        return
    if top < 0:
        yield 0
        return
    low, high = cofactors(node, top)
    yield from iterate(low, top - 1)
    for value in iterate(high, top - 1):
        yield 1 << top | value


@weak_cache
def node_count(node: BDDNode) -> int:
    seen, stack = set(), [node]
    while stack:
        if isinstance(current := stack.pop(), BDD) and current not in seen:
            seen.add(current)
            stack += current.left, current.right
    return len(seen)


def render_count(total: int) -> str:
    """`total`, written `2**k` when it is a large power of two."""
    if total >= 1024 and not total & (total - 1):
        return f"2**{total.bit_length() - 1}"
    return repr(total)


def pin(value: int, width: int) -> BDDNode:
    """`x == value`, over bits `[0, width)`."""
    node: BDDNode = ACCEPT
    for bit in range(width):
        node = BDD(bit, REJECT, node) if value >> bit & 1 else BDD(bit, node, REJECT)
    return node


def zeros(low: int, high: int) -> BDDNode:
    """Bits `[low, high)` all clear."""
    node: BDDNode = ACCEPT
    for bit in range(low, high):
        node = BDD(bit, node, REJECT)
    return node


def less_than(bound: int, width: int) -> BDDNode:
    """`x < bound`, over bits `[0, width)`."""
    if bound <= 0: return REJECT
    if bound >= 1 << width: return ACCEPT
    node: BDDNode = REJECT
    for bit in range(width):
        # Where `bound` has a set bit, a clear one in `x` settles it; where `bound`
        # has a clear bit, a set one rules `x` out. Otherwise the lower bits decide.
        node = BDD(bit, ACCEPT, node) if bound >> bit & 1 else BDD(bit, node, REJECT)
    return node
