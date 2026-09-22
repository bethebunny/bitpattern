from __future__ import annotations

import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Iterator, Self

__all__ = ["ACCEPT", "BDD", "BDDLeaf", "BDDNode", "REJECT"]

# The node algebra below is public to anyone extending the engine, but it is not
# re-exported: `count`, `index` and `size` are far too generic to hold names in a
# namespace someone imports from.

# Bound on memoised results per operation. BDD `apply` is exponential without
# memoisation; the cache is bounded so it cannot defeat the weak interning.
CACHE_SIZE = 1 << 16


class BDDNode:
    """A node in a reduced, ordered binary decision diagram.

    A diagram is a *predicate* over the bits of a non-negative integer: `bit` is
    the bit position tested here, `left` the branch taken when that bit is clear
    and `right` the branch taken when it is set. Bits no node tests are free, so
    a diagram is as large as the constraints it expresses rather than as large as
    the set it denotes.

    Nodes are interned, so structurally equal diagrams are the same object,
    equality is identity, and hashing is O(1) -- which matters, because interning
    keys contain child nodes.

    A predicate says nothing about how many bits are in play: `ACCEPT` denotes
    every integer. The sequence protocol here therefore reads a diagram over bits
    `[0, bit]` alone, and code wanting a wider universe passes the top bit to
    `count`, `index` and `iterate`. `IntSet` pairs a predicate with a width.
    """

    intern: ClassVar[weakref.WeakValueDictionary]

    bit: int

    def __invert__(self) -> BDDNode: raise NotImplementedError
    def __and__(self, other: BDDNode) -> BDDNode: raise NotImplementedError
    def __or__(self, other: BDDNode) -> BDDNode: raise NotImplementedError
    def __sub__(self, other: BDDNode) -> BDDNode: raise NotImplementedError

    def __len__(self) -> int: raise NotImplementedError
    def __getitem__(self, item: int) -> int: raise NotImplementedError

    def __iter__(self) -> Iterator[int]:
        return iterate(self, self.bit)


@dataclass(frozen=True, eq=False)
class BDDLeaf(BDDNode):
    value: bool

    intern: ClassVar[weakref.WeakValueDictionary[bool, BDDLeaf]] = weakref.WeakValueDictionary()

    @property
    def bit(self) -> int:
        return -1

    def __new__(cls, value: bool) -> Self:
        value = bool(value)
        if (obj := cls.intern.get(value)) is None:
            obj = object.__new__(cls)
            object.__setattr__(obj, "value", value)
            cls.intern[value] = obj
        return obj

    # Interned instances are constructed once, in __new__. Letting the dataclass
    # __init__ run again would rewrite the fields of the shared instance.
    def __init__(self, *_): pass

    def __invert__(self):
        return BDDLeaf(not self.value)

    def __bool__(self):
        return self.value

    def __and__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        return self and other

    def __or__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        return self or other

    def __sub__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        return self and ~other

    def __len__(self):
        return int(self.value)

    def __iter__(self):
        if self: yield 0

    def __getitem__(self, item):
        if not self or item not in (0, -1):
            raise IndexError(item)
        return 0

    def __repr__(self):
        return "ACCEPT" if self.value else "REJECT"

    def __reduce__(self):
        return BDDLeaf, (self.value,)


ACCEPT = BDDLeaf(True)
REJECT = BDDLeaf(False)


@dataclass(frozen=True, eq=False)
class BDD(BDDNode):
    bit: int
    left: BDDNode
    right: BDDNode

    intern: ClassVar[weakref.WeakValueDictionary[tuple, BDD]] = weakref.WeakValueDictionary()

    def __new__(cls, bit: int, left: BDDNode, right: BDDNode) -> BDDNode:
        if left is right: return left
        # Ordering is what makes the diagram reduced *and* ordered; out of order,
        # the scaling in `count` would shift a child's span off the end instead.
        if left.bit >= bit or right.bit >= bit:
            raise ValueError(
                f"children of bit {bit} must test lower bits, not {left.bit} and {right.bit}"
            )
        key = (bit, left, right)
        if (obj := cls.intern.get(key)) is None:
            obj = object.__new__(cls)
            object.__setattr__(obj, "bit", bit)
            object.__setattr__(obj, "left", left)
            object.__setattr__(obj, "right", right)
            cls.intern[key] = obj
        return obj

    # See BDDLeaf.__init__. Here it also guards the `left is right` reduction
    # above, which hands back a node that is still an instance of this class.
    def __init__(self, *_): pass

    # Rebuild through the constructor, so unpickling re-interns.
    def __reduce__(self):
        return BDD, (self.bit, self.left, self.right)

    @lru_cache(maxsize=CACHE_SIZE)
    def __invert__(self):
        return BDD(self.bit, ~self.left, ~self.right)

    @lru_cache(maxsize=CACHE_SIZE)
    def __and__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        match other:
            case BDDLeaf(_): return other & self
            case _ if self.bit < other.bit: return other & self
            case _ if self.bit > other.bit: return BDD(self.bit, self.left & other, self.right & other)
            case _:
                assert isinstance(other, BDD)
                return BDD(self.bit, self.left & other.left, self.right & other.right)

    @lru_cache(maxsize=CACHE_SIZE)
    def __or__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        match other:
            case BDDLeaf(_): return other | self
            case _ if self.bit < other.bit: return other | self
            case _ if self.bit > other.bit: return BDD(self.bit, self.left | other, self.right | other)
            case _:
                assert isinstance(other, BDD)
                return BDD(self.bit, self.left | other.left, self.right | other.right)

    def __sub__(self, other: BDDNode):
        if not isinstance(other, BDDNode): return NotImplemented
        return self & ~other

    def __len__(self):
        return size(self)

    def __repr__(self):
        return f"<BDD bit={self.bit}, {node_count(self)} nodes, {render_count(size(self))} members>"

    def __getitem__(self, item: int):
        total = size(self)
        if not -total <= item < total:
            raise IndexError(item)
        if item < 0: item += total
        if item < (left_size := count(self.left, self.bit - 1)):
            return index(self.left, self.bit - 1, item)
        return (1 << self.bit) | index(self.right, self.bit - 1, item - left_size)


@lru_cache(maxsize=CACHE_SIZE)
def node_count(node: BDDNode) -> int:
    """How many distinct nodes the diagram rooted here contains."""
    seen, stack = set(), [node]
    while stack:
        current = stack.pop()
        if isinstance(current, BDD) and id(current) not in seen:
            seen.add(id(current))
            stack += [current.left, current.right]
    return len(seen)


def render_count(total: int) -> str:
    """A cardinality, as a power of two where that is what it is.

    Every pattern-derived set has a power-of-two size, and `2**53` reads rather
    better than `9007199254740992`.
    """
    if total >= 1024 and not total & (total - 1):
        return f"2**{total.bit_length() - 1}"
    return repr(total)


@lru_cache(maxsize=CACHE_SIZE)
def size(node: BDDNode) -> int:
    """How many integers in `[0, 2 ** (node.bit + 1))` satisfy `node`.

    This, rather than `len`, is the primitive: `__len__` must fit a
    `Py_ssize_t`, and a 64-bit universe holds more members than that.
    """
    if not isinstance(node, BDD):
        return int(bool(node))
    return count(node.left, node.bit - 1) + count(node.right, node.bit - 1)


def count(node: BDDNode, top: int) -> int:
    """How many integers in `[0, 2 ** (top + 1))` satisfy `node`.

    Bits above `node.bit` and up to `top` are free, so each one doubles the
    count that `node` reports over its own narrower scope.
    """
    if node.bit > top:
        raise ValueError(f"node tests bit {node.bit}, above top bit {top}")
    return size(node) << (top - node.bit)


def index(node: BDDNode, top: int, item: int) -> int:
    """The `item`-th smallest integer in `[0, 2 ** (top + 1))` satisfying `node`.

    Free bits above `node.bit` are the *high* bits of the result, so ordering by
    them first and by the sub-diagram second is already ascending numeric order.
    """
    total = count(node, top)
    if not -total <= item < total:
        raise IndexError(item)
    if item < 0: item += total
    free, within = divmod(item, size(node))
    return (free << (node.bit + 1)) | node[within]


def iterate(node: BDDNode, top: int) -> Iterator[int]:
    """Every integer in `[0, 2 ** (top + 1))` satisfying `node`, ascending."""
    if node is REJECT:
        return
    if top < 0:
        yield 0
        return
    high = 1 << top
    # A free bit at `top` means the whole sub-diagram repeats, once with the bit
    # clear and once with it set.
    low, up = (node, node) if node.bit < top else (node.left, node.right)
    yield from iterate(low, top - 1)
    for value in iterate(up, top - 1):
        yield high | value


def pin(value: int, width: int) -> BDDNode:
    """The predicate satisfied by `value` alone, over bits `[0, width)`."""
    node: BDDNode = ACCEPT
    for bit in range(width):
        node = BDD(bit, REJECT, node) if (value >> bit) & 1 else BDD(bit, node, REJECT)
    return node


def zeros(low: int, high: int) -> BDDNode:
    """The predicate requiring bits `[low, high)` to be clear."""
    node: BDDNode = ACCEPT
    for bit in range(low, high):
        node = BDD(bit, node, REJECT)
    return node


def less_than(bound: int, width: int) -> BDDNode:
    """The predicate `x < bound`, over bits `[0, width)`.

    One node per bit, built from bit 0 up. At bit `i`: if `bound` has it set then
    a clear bit in `x` already settles the comparison and a set bit defers to the
    bits below; if `bound` has it clear then a set bit in `x` is already too big.
    """
    if bound <= 0: return REJECT
    if bound >= 1 << width: return ACCEPT
    node: BDDNode = REJECT
    for bit in range(width):
        node = BDD(bit, ACCEPT, node) if bound >> bit & 1 else BDD(bit, node, REJECT)
    return node
