"""Tuples of values from other codecs, side by side in one encoding."""

from __future__ import annotations

import functools
import itertools
import operator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeVar, TypeVarTuple, overload

from ..bdd import BDD, exists, shift
from ..intset import IntSet
from .codec import BDDSet, Codec

__all__ = ["Tuple"]

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
Ts = TypeVarTuple("Ts")


@dataclass(frozen=True, repr=False, init=False)
class Tuple(Codec[tuple[*Ts]]):
    """Tuples with a value from each field's codec, with the first field's bits on top.

    Sets of tuples iterate in order of their first field, then their second, and so
    on. `product` builds a set from a set for each field, and costs about as many
    nodes as they do together, however many tuples it holds. `project` goes the other
    way.

    >>> from bitpattern.codecs import ipv4, uint16
    >>> sockets = Tuple(ipv4, uint16)
    >>> private = sockets.product(ipv4.cidr("10.0.0.0/8"), uint16.range(1024, 65536))
    >>> private.size
    1082331758592
    >>> private[0]
    (IPv4Address('10.0.0.0'), 1024)
    >>> sockets.project(private, 1) == uint16.range(1024, 65536)
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
        return self.product(*(field.all for field in self.fields))

    def product(self, *sets: BDDSet[Any]) -> BDDSet[tuple[*Ts]]:
        """The tuples whose fields are in `sets`, one set for each field."""
        if [source.codec for source in sets] != list(self.fields):
            fields = ", ".join(map(repr, self.fields))
            raise TypeError(f"{self} needs a set for each of its fields, {fields}")
        # Each field's diagram sits above the next one's, so `&` just chains them.
        shifted = (
            shift(source.storage.bdd, offset)
            for source, offset in zip(sets, self._offsets)
        )
        bdd = functools.reduce(operator.and_, shifted, BDD.ACCEPT)
        return BDDSet(self, IntSet.from_bdd(bdd, self.width))

    def project(self, source: BDDSet[tuple[*Ts]], index: int) -> BDDSet[Any]:
        """The values `source` has in its field at `index`."""
        if source.codec != self:
            raise TypeError(f"{source.codec} sets aren't {self} sets")
        field, low = self.fields[index], self._offsets[index]
        # Forget the fields above and below this one, then move it down to bit 0.
        bdd = exists(source.storage.bdd, low + field.width, self.width)
        bdd = shift(exists(bdd, 0, low), -low)
        return BDDSet(field, IntSet.from_bdd(bdd, field.width))
