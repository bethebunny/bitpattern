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
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, TypeVarTuple, overload

from ..bdd import BDD, count_repr, freed_between, lshift
from ..dlpack import CPU, DType, exported, unsigned
from ..intset import IntSet, listed
from ..pattern import Pattern

if TYPE_CHECKING:
    from typing_extensions import CapsuleType

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
    ...         if not isinstance(value, str):
    ...             raise TypeError(f"{value!r} isn't a string")
    ...         if len(value) != 1 or not value.isascii():
    ...             raise ValueError(f"{value!r} isn't an ASCII character")
    ...         return ord(value)
    ...
    ...     def decode(self, bits: int) -> str:
    ...         return chr(bits)
    >>> ascii = Ascii(7, "ascii")
    >>> ascii.pattern("1?0.0001")  # case only changes bit 5
    BDDSet(ascii, ['A', 'a'])

    Codecs signal what they can't handle with:
      - `TypeError` from `encode`, for a value of the wrong type
      - `ValueError` from `encode`, for a value of the right type that isn't legal
      - `IndexError` from `decode`, for bits outside `all`, though `decode` doesn't
        have to check
    Codecs with illegal bit patterns override `all`, and `~` and `pattern` stay inside
    it.
    """

    width: int
    name: str

    def __repr__(self) -> str:
        return self.name

    def encode(self, value: object) -> int:
        """`value`'s bits. TypeError for the wrong type, and ValueError for an illegal
        value."""
        raise NotImplementedError

    def decode(self, bits: int) -> T:
        """Bits outside `all` may raise IndexError."""
        raise NotImplementedError

    def dtype(self) -> DType:
        """The C type that sets export their values as, eg. to `numpy.from_dlpack`.

        Codecs that encode values as a C type's bits (like int32) return that type, so a
        subclass that changes `encode` overrides `dtype` too. Others export as IntSets
        do.
        """
        return unsigned(self.width)

    @cached_property
    def all(self) -> BDDSet[T]:
        return BDDSet(self, IntSet.from_bdd(BDD.ACCEPT, self.width))

    @cached_property
    def none(self) -> BDDSet[T]:
        return BDDSet(self, IntSet.from_bdd(BDD.REJECT, self.width))

    def set(self, values: Iterable[T] = ()) -> BDDSet[T]:
        return BDDSet(self, IntSet(map(self.encode, values), width=self.width))

    def pattern(self, text: str) -> BDDSet[T]:
        """Values matching `text`, within `all`."""
        pattern = Pattern(text)
        if pattern.width != self.width:
            raise ValueError(f"pattern is {pattern.width} bits, not {self.width}")
        return BDDSet(self, pattern) & self.all


class OrderedCodec(Codec[T]):
    """A codec whose values have an order, with `min`, `max` and `range`.

    The defaults assume encodings sort the same as their values. Codecs that encode
    in another order, like `Int` and `Float`, override the defaults.
    """

    @property
    def min(self) -> T:
        return self.all[0]

    @property
    def max(self) -> T:
        return self.all[-1]

    def range(self, start: T | None = None, stop: T | None = None) -> BDDSet[T]:
        """Values in `[start, stop)`. `None` is unbounded."""
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

    Set operators and comparisons only accept sets of the same codec, and raise
    TypeError for anything else (convert with `set(s)` or `codec.set(other)`).
    """

    codec: Codec[T]
    storage: IntSet

    def __post_init__(self) -> None:
        if self.storage.width != self.codec.width:
            raise ValueError(
                f"{self.codec} needs {self.codec.width} bits, not {self.storage.width}"
            )

    def _storage(self, other: object) -> IntSet:
        # `is` first, since it's quicker.
        if isinstance(other, BDDSet) and (
            other.codec is self.codec or other.codec == self.codec
        ):
            return other.storage
        # Not NotImplemented, since IntSet's reflected operators would enumerate the set.
        raise TypeError(f"{self.codec} sets only combine with each other")

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
        except (TypeError, ValueError):  # not something this codec can encode
            return False

    def __iter__(self) -> Iterator[T]:
        return map(self.codec.decode, self.storage)

    def __reversed__(self) -> Iterator[T]:
        return map(self.codec.decode, reversed(self.storage))

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType:
        """The values as a DLPack tensor of `codec.dtype()`. Values wider than an
        unsigned type are rows of words."""
        return exported(
            self.storage.bdd,
            width=self.codec.width,
            dtype=self.codec.dtype(),
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        return (CPU, 0)

    @overload
    def __getitem__(self, item: int) -> T: ...
    @overload
    def __getitem__(self, item: slice) -> BDDSet[T]: ...
    def __getitem__(self, item: int | slice) -> T | BDDSet[T]:
        if isinstance(item, slice):
            return BDDSet(self.codec, self.storage[item])
        return self.codec.decode(self.storage[item])

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        try:
            bits = self.codec.encode(value)
        except TypeError as e:  # as for list.index
            raise ValueError(f"{value!r} is not in the set") from e
        return self.storage.index(bits, start, stop)

    def count(self, value: object) -> int:
        return int(value in self)

    def choice(self, *, rng: random.Random | None = None) -> T:
        """Like `random.choice`, but at any size."""
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
        """`product(self, other)`."""
        if not isinstance(other, BDDSet):
            return NotImplemented
        return product(self, other)

    def isdisjoint(self, other: Iterable[T]) -> bool:
        return self.storage.isdisjoint(self._storage(other))

    def __repr__(self) -> str:
        if (size := self.size) <= 10:
            return f"BDDSet({self.codec}, [{listed(self)}])"
        head, tail = listed(itertools.islice(self, 4)), listed([self[-2], self[-1]])
        return f"BDDSet({self.codec}, [{head}, ..., {tail}], size={count_repr(size)})"


@dataclass(frozen=True, repr=False, init=False)
class Tuple(Codec[tuple[*Ts]]):
    """Tuples with a value from each field's codec, with the first field's bits on top.

    Sets of tuples sort by their first field, then their second, and so on. `product`
    makes them (`a * b` is `product(a, b)`), with about as many nodes as its sets
    combined. `project` gets one field's values.

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
        """Each field's lowest bit."""
        widths = [field.width for field in reversed(self.fields[1:])]
        return tuple(itertools.accumulate(widths, initial=0))[::-1]

    @cached_property
    def _layout(self) -> tuple[tuple[Codec[Any], int, int], ...]:
        """Each field with its lowest bit and its mask."""
        masks = [(1 << field.width) - 1 for field in self.fields]
        return tuple(zip(self.fields, self._offsets, masks))

    def encode(self, value: object) -> int:
        if not isinstance(value, tuple):
            raise TypeError(f"{self} can't encode {value!r}")
        if len(value) != len(self.fields):
            raise ValueError(f"{self} can't encode {value!r}")
        bits = 0
        try:
            for field, item in zip(self.fields, value):
                bits = bits << field.width | field.encode(item)
        except StopIteration as e:
            # As a generator would, so a field's StopIteration can't quietly end an
            # iteration (eg. in `set`).
            raise RuntimeError(f"{self} raised StopIteration") from e
        return bits

    def decode(self, bits: int) -> tuple[*Ts]:
        # A generator, so a field's StopIteration becomes a RuntimeError.
        return tuple(
            field.decode(bits >> low & mask) for field, low, mask in self._layout
        )

    @cached_property
    def all(self) -> BDDSet[tuple[*Ts]]:
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
    # Each diagram sits above the next, so `&` chains them.
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
    bdd = freed_between(
        source.storage.bdd, low_bit=low + field.width, high_bit=codec.width
    )
    bdd = lshift(freed_between(bdd, low_bit=0, high_bit=low), -low)
    return BDDSet(field, IntSet.from_bdd(bdd, field.width))
