"""UUIDs as their 128 bits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from uuid import UUID

from ..intset import IntSet
from ..pattern import diagram
from .codec import BDDSet, OrderedCodec

__all__ = ["UUIDCodec", "uuid128"]


@dataclass(frozen=True, repr=False)
class UUIDCodec(OrderedCodec[UUID]):
    """`uuid.UUID`s, in Python's sort order.

    `encode` accepts UUIDs or their strings. Patterns have one quartet per hex digit
    of `xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx`. RFC 9562 puts the version in M and the
    variant in the top bits of N.
    """

    def encode(self, value: object) -> int:
        match value:
            case UUID():
                return value.int
            case str():
                return UUID(value).int  # raises ValueError for a malformed string
        raise TypeError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> UUID:
        try:
            return UUID(int=bits)
        except ValueError as e:  # wider than 128 bits
            raise IndexError(f"{self} has no value encoded as {bits}") from e

    def _rfc(self, version: str) -> BDDSet[UUID]:
        """The RFC 9562 variant, with `version` as M's bits, eg. `0100` or `????`."""
        # M is the 13th hex digit and N the 17th, so they start 48 and 64 bits in.
        bits = "?" * 48 + version + "?" * 12 + "10" + "?" * 62
        return BDDSet(self, IntSet.from_bdd(diagram(bits), self.width))

    @cached_property
    def rfc9562(self) -> BDDSet[UUID]:
        """UUIDs in RFC 9562's variant, where N is 8, 9, a or b. `uuid` calls the
        variant `RFC_4122`."""
        return self._rfc("????")

    def version(self, n: int) -> BDDSet[UUID]:
        """UUIDs of RFC 9562 version `n`, from 1 to 8."""
        if not 1 <= n <= 8:
            raise ValueError(f"UUID versions go from 1 to 8, not {n}")
        return self._rfc(f"{n:04b}")


uuid128 = UUIDCodec(128, "uuid128")
