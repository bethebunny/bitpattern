"""Enum members as their index, and flags as their own bits."""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from enum import Enum, Flag
from functools import cached_property
from typing import TypeVar

from ..bdd import BDD, diagram
from ..intset import IntSet
from .codec import BDDSet, Codec

__all__ = ["Flags", "Members"]

E = TypeVar("E", bound=Enum)
F = TypeVar("F", bound=Flag)


@dataclass(frozen=True, repr=False, init=False)
class Members(Codec[E]):
    """An Enum's members, in definition order.

    Sets export each member's index, not its value, even for an IntEnum.

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
        if not isinstance(value, self.enum):
            raise TypeError(f"{self} can't encode {value!r}")
        # A Flag's combinations are instances too, but aren't members.
        if (index := self._indices.get(value)) is None:
            raise ValueError(f"{self} can't encode {value!r}")
        return index

    def decode(self, bits: int) -> E:
        return self._members[bits]  # raises IndexError past the last member

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
        # iter(Flag) doesn't include multi-bit members, use __members__
        width = max(
            (member.value for member in enum.__members__.values()), default=0
        ).bit_length()
        vars(self).update(width=width, name=enum.__name__, enum=enum)

    @cached_property
    def _single(self) -> int:
        """The bits that are each a flag of their own."""
        single = 0
        for member in self.enum.__members__.values():
            if member.value.bit_count() == 1:
                single |= member.value
        return single

    def encode(self, value: object) -> int:
        if not isinstance(value, self.enum):
            raise TypeError(f"{self} can't encode {value!r}")
        # Python Flag can make values that aren't legal combinations. A value whose bits
        # are each a flag is one, and anything else has to be checked.
        if (bits := value._value_) & ~self._single and bits not in self.all.storage:
            raise ValueError(f"{self} can't encode {value!r}")
        return bits

    def decode(self, bits: int) -> F:
        try:
            return self.enum(bits)
        except ValueError as e:  # a strict Flag refuses bits outside every flag
            raise IndexError(f"{self} has no value encoded as {bits}") from e

    def _glob(self, flags: int, bit: str) -> BDDSet[F]:
        """Bit patterns with `bit` at each bit of `flags`, and `?` elsewhere."""
        bits = f"{flags:0{self.width}b}".replace("0", "?").replace("1", bit)
        return BDDSet(self, IntSet.from_bdd(diagram(bits), self.width))

    @cached_property
    def all(self) -> BDDSet[F]:
        """Every combination of the flags, including none."""
        # iter(Flag) doesn't include multi-bit members, use __members__
        flags = [member.value for member in self.enum.__members__.values()]
        # Flags can have some weird properties:
        #   - Flags aren't required to be dense, ie. they can skip bits
        #   - Flags aren't required to be powers of 2, ie. they can set multiple bits
        #   - A flag can include another flag's bits, eg. ssl.VerifyFlags'
        #     VERIFY_CRL_CHECK_CHAIN (0xc) includes VERIFY_CRL_CHECK_LEAF (0x4), so
        #     setting CHAIN always sets LEAF
        # Algorithm, treating values and flags as sets of bits:
        #   - Start with every bit pattern.
        #   - For each bit `b`, remove each `x` with `b ∈ x` where no flag `f` with
        #     `b ∈ f` has `f ⊆ x`.
        # Proof that this leaves exactly the unions of flags:
        #   - If `x` is a union of flags, each `b ∈ x` is in one of those flags `f`,
        #     and `f ⊆ x`. Thus no step removes `x`.
        #   - If `x` isn't a union of flags, let `u` be the union of every flag
        #     `f ⊆ x`. `u` is a union of flags and `x` isn't, so `u != x`. Since
        #     `u ⊂ x`, some `b ∈ x` has `b ∉ u`. No flag `f` with `b ∈ f` has
        #     `f ⊆ x`, since then `f ⊆ u` and `b ∈ u`. Thus the step for `b` removes
        #     `x`.
        combinations = BDDSet(self, IntSet.from_bdd(BDD.ACCEPT, self.width))
        for at in range(self.width):
            # Each bit is clear, or part of a flag that's wholly set.
            # The inverse of this set is the set of values where the bit is set
            # but no flag containing the bit is wholly set.
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
