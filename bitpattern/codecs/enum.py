"""Enum members as their index, and flags as their own bits."""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from enum import Enum, Flag
from functools import cached_property
from typing import TypeVar

from ..bdd import BDD
from ..intset import IntSet
from ..pattern import diagram
from .codec import BDDSet, Codec

__all__ = ["Flags", "Members"]

E = TypeVar("E", bound=Enum)
F = TypeVar("F", bound=Flag)


@dataclass(frozen=True, repr=False, init=False)
class Members(Codec[E]):
    """An Enum's members, in the order they're defined.

    >>> from enum import Enum
    >>> class Color(Enum):
    ...     RED = "r"
    ...     GREEN = "g"
    ...     BLUE = "b"
    >>> colors = Members(Color)
    >>> colors.width
    2
    >>> ~colors.set([Color.GREEN])
    BDDSet(Color, [<Color.RED: 'r'>, <Color.BLUE: 'b'>])
    """

    enum: type[E]

    def __init__(self, enum: type[E]) -> None:
        width = max(len(enum) - 1, 0).bit_length()
        vars(self).update(width=width, name=enum.__name__, enum=enum)

    @cached_property
    def _members(self) -> tuple[E, ...]:
        return tuple(self.enum)

    @cached_property
    def _indices(self) -> dict[E, int]:
        return {member: at for at, member in enumerate(self._members)}

    def encode(self, value: object) -> int:
        # A Flag's combinations are instances too, but they aren't members.
        if isinstance(value, self.enum) and value in self._indices:
            return self._indices[value]
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> E:
        return self._members[bits]

    @cached_property
    def all(self) -> BDDSet[E]:
        """Every member."""
        return BDDSet(self, IntSet.range(0, len(self._members), width=self.width))


@dataclass(frozen=True, repr=False, init=False)
class Flags(Codec[F]):
    """Combinations of a Flag's flags, as their own bits.

    Sets iterate in order of value.

    >>> from enum import Flag
    >>> class Perm(Flag):
    ...     R = 4
    ...     W = 2
    ...     X = 1
    >>> perms = Flags(Perm)
    >>> perms.pattern("1?0")
    BDDSet(Perm, [<Perm.R: 4>, <Perm.R|W: 6>])
    >>> perms.having(Perm.W) == perms.pattern("?1?")
    True
    """

    enum: type[F]

    def __init__(self, enum: type[F]) -> None:
        flags = (member.value for member in enum.__members__.values())
        width = max(flags, default=0).bit_length()
        vars(self).update(width=width, name=enum.__name__, enum=enum)

    def encode(self, value: object) -> int:
        # Flag can make values that aren't combinations, eg. IntFlag keeps stray bits.
        if isinstance(value, self.enum) and value.value in self.all.storage:
            return value.value
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> F:
        return self.enum(bits)

    def _glob(self, flags: int, bit: str) -> BDDSet[F]:
        """Bit patterns with `bit` at each bit of `flags`, and `?` elsewhere."""
        bits = f"{flags:0{self.width}b}".replace("0", "?").replace("1", bit)
        return BDDSet(self, IntSet.from_bdd(diagram(bits), self.width))

    @cached_property
    def all(self) -> BDDSet[F]:
        """Every combination of the flags, including none."""
        flags = {member.value for member in self.enum.__members__.values()}
        combinations = BDDSet(self, IntSet.from_bdd(BDD.ACCEPT, self.width))
        for at in range(self.width):
            # Each bit is clear, or part of a flag that's wholly set.
            whole = (self._glob(flag, "1") for flag in flags if flag >> at & 1)
            clear = self._glob(1 << at, "0")
            combinations &= functools.reduce(operator.or_, whole, clear)
        return combinations

    def having(self, flags: F) -> BDDSet[F]:
        """Every combination with all of `flags` set."""
        return self._glob(self.encode(flags), "1") & self.all

    def lacking(self, flags: F) -> BDDSet[F]:
        """Every combination with none of `flags` set."""
        return self._glob(self.encode(flags), "0") & self.all
