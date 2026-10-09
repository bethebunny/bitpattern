"""Codecs and their sets.

A Codec encodes values as integers. A BDDSet is a set of one codec's values. Tuple is
a codec of tuples, with a codec per field.
"""

from __future__ import annotations

import functools
import itertools
import operator
import random
from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, Literal, TypeVar, TypeVarTuple, overload

from ..bdd import BDD, freed_between, lshift, render_count
from ..intset import IntSet
from ..pattern import Pattern

__all__ = ["BDDSet", "Codec", "OrderedCodec", "Tuple", "product", "project"]

T = TypeVar("T")
U = TypeVar("U")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
Ts = TypeVarTuple("Ts")


@dataclass(frozen=True, repr=False)
class Codec(Generic[T]):
    """A fixed-width encoding of values as non-negative integers.

    A codec's sets are BDDSets, ie. IntSets of its encodings. Patterns match the
    encodings' bits. A codec instance is also a namespace for important sets of that
    type, like `float64.subnormal` or `ipv4.cidr("10.0.0.0/8")`.

    Codec subclasses implement `encode` and `decode`:

    >>> class Ascii(Codec[str]):
    ...     def encode(self, value: object) -> int:
    ...         if isinstance(value, str) and len(value) == 1 and value.isascii():
    ...             return ord(value)
    ...         raise ValueError(f"{value!r} isn't an ASCII character")
    ...
    ...     def decode(self, bits: int) -> str:
    ...         return chr(bits)
    >>> ascii = Ascii(7, "ascii")
    >>> ascii.pattern("1?0.0001")  # case only changes bit 5
    BDDSet(ascii, ['A', 'a'])

    If a type has illegal bit patterns then `encode` should raise a `ValueError`.
    In this case, the Codec should override `all` to return the legal pattern
    domain for the type. `~` and `pattern` then stay inside it.
    """

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
        """Every value. That's every bit pattern unless a codec overrides it."""
        return BDDSet(self, IntSet.from_bdd(BDD.ACCEPT, self.width))

    @cached_property
    def none(self) -> BDDSet[T]:
        """The empty set."""
        return BDDSet(self, IntSet.from_bdd(BDD.REJECT, self.width))

    def set(self, values: Iterable[T] = ()) -> BDDSet[T]:
        """Constructs a BDDSet from the input values."""
        return BDDSet(self, IntSet(map(self.encode, values), width=self.width))

    def pattern(self, text: str) -> BDDSet[T]:
        """Constructs a BDDSet of values matching the pattern. Bit patterns outside
        `all` are left out."""
        pattern = Pattern(text)
        if pattern.width != self.width:
            raise ValueError(f"pattern is {pattern.width} bits, not {self.width}")
        return BDDSet(self, pattern) & self.all


class OrderedCodec(Codec[T]):
    """A codec whose values have an order, with `min`, `max` and `range`.

    The defaults assume encodings sort the same as their values. Codecs that encode
    in another order, like `Int` and `Float`, override them.
    """

    @property
    def min(self) -> T:
        """The smallest value."""
        return self.all[0]

    @property
    def max(self) -> T:
        """The largest value."""
        return self.all[-1]

    def range(self, start: T | None = None, stop: T | None = None) -> BDDSet[T]:
        """Values in `[start, stop)`. A bound of `None` leaves that end open."""
        low = 0 if start is None else self.encode(start)
        high = 1 << self.width if stop is None else self.encode(stop)
        return BDDSet(self, IntSet.range(low, high, width=self.width)) & self.all


@dataclass(frozen=True, repr=False)
class BDDSet(AbstractSet[T], Sequence[T], Generic[T]):
    """An ordered set of a codec's values kept as the IntSet of their bits.

    >>> from bitpattern.codecs import float64
    >>> normal = float64.finite - float64.subnormal - float64.zeros
    >>> normal.size
    18428729675200069632
    >>> normal[0]
    2.2250738585072014e-308
    >>> 1.0 in normal
    True
    >>> float("inf") in normal
    False

    Python isn't really designed for data structures larger than memory. In
    particular many operations will fail if `__len__` returns a number >= 2**63.
    When working with very large sets:

    - Use `s.size` instead of `len(s)`
    - Use `s.choice()` instead of `random.choice(s)` or `random.sample(s, k)`
    - Use `s[0]` and `s[-1]` instead of `min(s)` and `max(s)`, where encodings sort
      like their values (not signed `Int`s or `Float`s)
    - Use `bitpattern.strategies` instead of hypothesis's `sampled_from(s)`

    For width w and diagrams of |a| and |b| nodes:

    - Most operations are O(w), however many members there are
    - `&`, `|`, `-` and `^` are O(|a| * |b|)
    - `~` is O(|a|)

    A diagram has at most `size * w` nodes and usually far fewer, eg. a pattern has
    at most one node per bit.

    Set operators and comparisons only accept sets of the same codec. Anything else
    raises TypeError, including plain sets. Convert this set via `set(bddset)` or
    the other via `codec.set(other)` for comparison.
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
            # Raise rather than return NotImplemented. IntSet's reflected operators
            # would try to enumerate the set.
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
        """Every other value in `codec.all`."""
        return self.codec.all - self

    def __le__(self, other: object) -> bool:
        return self.storage <= self._storage(other)

    def __lt__(self, other: object) -> bool:
        return self.storage < self._storage(other)

    def __ge__(self, other: object) -> bool:
        return self.storage >= self._storage(other)

    def __gt__(self, other: object) -> bool:
        return self.storage > self._storage(other)

    def __mul__(self, other: BDDSet[U]) -> BDDSet[tuple[T, U]]:
        """Pairs from this set and `other`, ie. `product(self, other)`."""
        if not isinstance(other, BDDSet):
            return NotImplemented
        return product(self, other)

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


@dataclass(frozen=True, repr=False, init=False)
class Tuple(Codec[tuple[*Ts]]):
    """Tuples with a value from each field's codec, with the first field's bits on top.

    Sets of tuples iterate in order of their first field, then their second, and so
    on. `product(a, b, c)` makes them, and `a * b` is `product(a, b)`. A product has
    about as many nodes as its sets combined. `project` gets the values of one field.

    >>> from bitpattern.codecs import ipv4, uint16
    >>> private = ipv4.cidr("10.0.0.0/8") * uint16.range(1024, 65536)
    >>> private.size
    1082331758592
    >>> private[0]
    (IPv4Address('10.0.0.0'), 1024)
    >>> project(private, 1) == uint16.range(1024, 65536)
    True
    """

    fields: tuple[Codec[Any], ...]

    @overload
    def __init__(self: Tuple[A], a: Codec[A], /) -> None: ...
    @overload
    def __init__(self: Tuple[A, B], a: Codec[A], b: Codec[B], /) -> None: ...
    @overload
    def __init__(
        self: Tuple[A, B, C], a: Codec[A], b: Codec[B], c: Codec[C], /
    ) -> None: ...
    @overload
    def __init__(
        self: Tuple[A, B, C, D], a: Codec[A], b: Codec[B], c: Codec[C], d: Codec[D], /
    ) -> None: ...
    @overload
    def __init__(self: Tuple[*tuple[Any, ...]], *fields: Codec[Any]) -> None: ...
    def __init__(self, *fields: Codec[Any]) -> None:
        width = sum(field.width for field in fields)
        name = f"tuple[{', '.join(map(repr, fields))}]"
        vars(self).update(width=width, name=name, fields=fields)

    @cached_property
    def _offsets(self) -> tuple[int, ...]:
        """Where each field's lowest bit is."""
        widths = [field.width for field in reversed(self.fields[1:])]
        return tuple(itertools.accumulate(widths, initial=0))[::-1]

    def encode(self, value: object) -> int:
        if not isinstance(value, tuple) or len(value) != len(self.fields):
            raise ValueError(f"{self} can't encode {value!r}")
        return sum(
            field.encode(item) << offset
            for field, item, offset in zip(self.fields, value, self._offsets)
        )

    def decode(self, bits: int) -> tuple[*Ts]:
        return tuple(
            field.decode(bits >> offset & ((1 << field.width) - 1))
            for field, offset in zip(self.fields, self._offsets)
        )

    @cached_property
    def all(self) -> BDDSet[tuple[*Ts]]:
        """Every tuple of the fields' values."""
        return product(*(field.all for field in self.fields))


@overload
def product(a: BDDSet[A], /) -> BDDSet[tuple[A]]: ...
@overload
def product(a: BDDSet[A], b: BDDSet[B], /) -> BDDSet[tuple[A, B]]: ...
@overload
def product(a: BDDSet[A], b: BDDSet[B], c: BDDSet[C], /) -> BDDSet[tuple[A, B, C]]: ...
@overload
def product(
    a: BDDSet[A], b: BDDSet[B], c: BDDSet[C], d: BDDSet[D], /
) -> BDDSet[tuple[A, B, C, D]]: ...
@overload
def product(*sets: BDDSet[Any]) -> BDDSet[tuple[Any, ...]]: ...
def product(*sets: BDDSet[Any]) -> BDDSet[tuple[Any, ...]]:
    """The tuples with a value from each of `sets`, like `itertools.product`."""
    codec = Tuple(*(source.codec for source in sets))
    # Each set's diagram sits above the next one's, so `&` just chains them.
    shifted = (
        lshift(source.storage.bdd, offset)
        for source, offset in zip(sets, codec._offsets)
    )
    bdd = functools.reduce(operator.and_, shifted, BDD.ACCEPT)
    return BDDSet(codec, IntSet.from_bdd(bdd, codec.width))


# Types can't index into a tuple of any length, so each index gets its own overload.
@overload
def project(
    source: BDDSet[tuple[A, *tuple[Any, ...]]], index: Literal[0]
) -> BDDSet[A]: ...
@overload
def project(
    source: BDDSet[tuple[Any, A, *tuple[Any, ...]]], index: Literal[1]
) -> BDDSet[A]: ...
@overload
def project(
    source: BDDSet[tuple[Any, Any, A, *tuple[Any, ...]]], index: Literal[2]
) -> BDDSet[A]: ...
@overload
def project(
    source: BDDSet[tuple[Any, Any, Any, A, *tuple[Any, ...]]], index: Literal[3]
) -> BDDSet[A]: ...
@overload
def project(
    source: BDDSet[tuple[*tuple[Any, ...], A]], index: Literal[-1]
) -> BDDSet[A]: ...
@overload
def project(source: BDDSet[tuple[Any, ...]], index: int) -> BDDSet[Any]: ...
def project(source: BDDSet[tuple[Any, ...]], index: int) -> BDDSet[Any]:
    """The values that tuples in `source` have at `index`."""
    if not isinstance(codec := source.codec, Tuple):
        raise TypeError(f"{codec} sets aren't sets of tuples")
    field, low = codec.fields[index], codec._offsets[index]
    # Free the other fields' bits, then move this field down to bit 0.
    bdd = freed_between(
        source.storage.bdd, low_bit=low + field.width, high_bit=codec.width
    )
    bdd = lshift(freed_between(bdd, low_bit=0, high_bit=low), -low)
    return BDDSet(field, IntSet.from_bdd(bdd, field.width))
