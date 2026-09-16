from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import ClassVar, TypeVar, Self, Iterable

Key = TypeVar("Key")

class BDDNode:
    intern: ClassVar[weakref.WeakValueDictionary[Key, Self]] = weakref.WeakValueDictionary()

    bit: int
    def __invert__(self) -> BDDNode: raise NotImplementedError
    def __and__(self, other: BDDNode) -> BDDNode: raise NotImplementedError
    def __or__(self, other: BDDNode) -> BDDNode: raise NotImplementedError
    def __sub__(self, other: BDDNode) -> BDDNode: raise NotImplementedError

    def __len__(self) -> int: raise NotImplementedError
    def __getitem__(self, item: int) -> int: raise NotImplementedError

@dataclass(frozen=True)
class BDDLeaf(BDDNode):
    value: bool

    @property
    def bit(self) -> int:
        return -1

    def __new__(cls, value: bool):
        if (obj := cls.intern.get(value)) is None:
            obj = object.__new__(cls)
            cls.intern[value] = obj
        return obj

    def __invert__(self):
        return BDDLeaf(not self.value)

    def __bool__(self):
        return self.value

    def __and__(self, other: BDDNode):
        return self and other

    def __or__(self, other: BDDNode):
        return self or other

    def __sub__(self, other: BDDNode):
        return self and ~other

    def __len__(self):
        return bool(self.value)

    def __iter__(self):
        if self: yield 0

    def __getitem__(self, item):
        assert self and item == 0
        return 0


ACCEPT = BDDLeaf(True)
REJECT = BDDLeaf(False)


@dataclass(frozen=True)
class BDD(BDDNode):
    bit: int
    left: BDDNode
    right: BDDNode

    def __new__(cls, bit: int, left: BDDNode, right: BDDNode):
        if left is right: return left
        key = (bit, left, right)
        if (obj := cls.intern.get(key)) is None:
            obj = object.__new__(cls)
            cls.intern[key] = obj
        return obj

    def __invert__(self):
        return type(self)(self.bit, ~self.left, ~self.right)

    def __and__(self, other: BDDNode):
        match other:
            case BDDLeaf(_): return other & self
            case _ if self.bit < other.bit: return other & self
            case _ if self.bit > other.bit: return BDD(self.bit, self.left & other, self.right & other)
            case _:
                assert isinstance(other, BDD)
                return BDD(self.bit, self.left & other.left, self.right & other.right)

    def __or__(self, other: BDDNode):
        match other:
            case BDDLeaf(_): return other | self
            case _ if self.bit < other.bit: return other | self
            case _ if self.bit > other.bit: return BDD(self.bit, self.left | other, self.right | other)
            case _:
                assert isinstance(other, BDD)
                return BDD(self.bit, self.left | other.left, self.right | other.right)

    def __sub__(self, other: BDDNode):
        return self & ~other

    def __len__(self):
        return self.child_len(self.left) + self.child_len(self.right)

    def child_len(self, child: BDDNode):
        return len(child) << self.bit >> (child.bit + 1)

    def child_idx(self, idx: int, child: BDDNode):
        prefix_idx, child_idx = divmod(idx, len(child))
        return (prefix_idx << (child.bit + 1)) | child[child_idx]

    def __getitem__(self, item: int):
        if not -len(self) <= item < len(self):
            raise IndexError(item)
        if item < 0: item += len(self)
        if item < (left_children := self.child_len(self.left)):
            return self.child_idx(item, self.left)
        return (1 << self.bit) | self.child_idx(item - left_children, self.right)


class IntSet:
    def __init__(self, iterable: Iterable[int] = ()):
        self.bdd: BDDNode = REJECT

        if isinstance(iterable, BDDNode):
            self.bdd = iterable
            return

        for v in iterable:
            self.add(v)

    @classmethod
    def from_bdd(cls, bdd: BDDNode) -> Self:
        self = cls()
        self.bdd = bdd

    def add(self, v: int):
        ...

    def __contains__(self, v: int) -> bool:
        ...

    def __iter__(self):
        yield from self.bdd

    def __len__(self):
        return len(self.bdd)

    def __and__(self, other: Iterable[int]):
        other = IntSet(other)
        return IntSet.from_bdd(self.bdd & other.bdd)

    def __or__(self, other: Iterable[int]):
        other = IntSet(other)
        return IntSet.from_bdd(self.bdd | other.bdd)

    def __sub__(self, other: Iterable[int]):
        other = IntSet(other)
        return IntSet.from_bdd(self.bdd - other.bdd)

    def __getitem__(self, item: int) -> int:
        return self.bdd[item]
