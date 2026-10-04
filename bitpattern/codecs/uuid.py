"""UUIDs, as their 128 bits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from uuid import UUID

from ..intset import IntSet
from ..pattern import diagram
from .codec import BDDSet, Codec

__all__ = ["UUIDCodec", "uuid128"]


@dataclass(frozen=True, repr=False)
class UUIDCodec(Codec[UUID]):
    """`uuid.UUID`s, as the bits of `UUID.int`, which is also how UUIDs sort.

    A UUID is written `xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx`, one hex digit per
    quartet of a pattern. RFC 9562 puts the version in M and the variant in the top
    bits of N.
    """

    def encode(self, value: object) -> int:
        match value:
            case UUID():
                return value.int
            case str():
                return UUID(value).int  # raises ValueError if it's malformed
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> UUID:
        return UUID(int=bits)

    def _rfc(self, version: str) -> BDDSet[UUID]:
        """The RFC 9562 variant, with `version` as M's bits, eg. `0100` or `????`."""
        # M is the 13th hex digit and N the 17th, so they start 48 and 64 bits in.
        bits = "?" * 48 + version + "?" * 12 + "10" + "?" * 62
        return BDDSet(self, IntSet.from_bdd(diagram(bits), self.width))

    @cached_property
    def rfc9562(self) -> BDDSet[UUID]:
        """UUIDs in the variant RFC 9562 defines, where N is 8, 9, a or b. The `uuid`
        module calls it `RFC_4122`."""
        return self._rfc("????")

    def version(self, n: int) -> BDDSet[UUID]:
        """UUIDs of RFC 9562 version `n`, from 1 to 8."""
        if not 1 <= n <= 8:
            raise ValueError(f"UUID versions go from 1 to 8, not {n}")
        return self._rfc(f"{n:04b}")

    @cached_property
    def nil(self) -> BDDSet[UUID]:
        """`00000000-0000-0000-0000-000000000000`, which isn't in `rfc9562`."""
        return self.set([UUID(int=0)])

    @cached_property
    def max(self) -> BDDSet[UUID]:
        """`ffffffff-ffff-ffff-ffff-ffffffffffff`, which isn't in `rfc9562`."""
        return self.set([UUID(int=(1 << self.width) - 1)])


uuid128 = UUIDCodec(128, "uuid128")
