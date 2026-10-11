"""MAC addresses, as their raw bits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

from ..intset import IntSet
from ..pattern import diagram
from .codec import BDDSet, OrderedCodec

__all__ = ["MAC", "mac"]

ADDRESS = re.compile(
    r"""
      (?:[0-9a-f]{2}:){5}[0-9a-f]{2}    # 01:23:45:67:89:ab
    | (?:[0-9a-f]{2}-){5}[0-9a-f]{2}    # 01-23-45-67-89-ab
    | (?:[0-9a-f]{4}\.){2}[0-9a-f]{4}   # Cisco's 0123.4567.89ab
    """,
    re.IGNORECASE | re.VERBOSE,
)
# The first three octets of an address.
OUI = re.compile(
    r"""
      (?:[0-9a-f]{2}:){2}[0-9a-f]{2}    # 00:1a:2b
    | (?:[0-9a-f]{2}-){2}[0-9a-f]{2}    # 00-1a-2b
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, repr=False)
class MAC(OrderedCodec[str]):
    """48-bit MAC addresses (EUI-48), like `01:23:45:67:89:ab`.

    MAC sets have the same ordering as their string representations.
    The two least significant bits of the first octet are flags:
      - bit 0 is I/G, set only on `multicast` addresses
      - bit 1 is U/L, set only on `local` addresses

    Supported formats:
      - colon separated, eg. `01:23:45:67:89:ab`
      - hyphen separated, eg. `01-23-45-67-89-ab`
      - Cisco, eg. `0123.4567.89ab`
    Hex digits may be lower or uppercase. Addresses always decode to
    lowercase with colons.
    """

    def encode(self, value: object) -> int:
        if isinstance(value, str) and ADDRESS.fullmatch(value):
            return int(re.sub("[.:-]", "", value), 16)
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> str:
        return bits.to_bytes(6).hex(":")

    def _prefix(self, bits: str) -> BDDSet[str]:
        """Addresses whose leading bits match `bits`, eg. `???????1` is `multicast`."""
        bdd = diagram(bits.ljust(self.width, "?"))
        return BDDSet(self, IntSet.from_bdd(bdd, self.width))

    @cached_property
    def multicast(self) -> BDDSet[str]:
        """Group addresses (I/G bit set)."""
        return self._prefix("???????1")

    @cached_property
    def unicast(self) -> BDDSet[str]:
        """Individual addresses (I/G bit clear)."""
        return self._prefix("???????0")

    @cached_property
    def local(self) -> BDDSet[str]:
        """Locally administered addresses (U/L bit set)."""
        return self._prefix("??????1?")

    @cached_property
    def universal(self) -> BDDSet[str]:
        """Universally administered addresses (U/L bit clear)."""
        return self._prefix("??????0?")

    @cached_property
    def broadcast(self) -> BDDSet[str]:
        """`ff:ff:ff:ff:ff:ff`, also in `multicast` and `local`."""
        return self.set(["ff:ff:ff:ff:ff:ff"])

    def oui(self, prefix: str) -> BDDSet[str]:
        """Addresses under a 24-bit OUI (Organizationally Unique Identifier) eg. `mac.oui("00:1a:2b")`."""
        if not OUI.fullmatch(prefix):
            raise ValueError(
                f"bad OUI {prefix!r}. OUIs are three octets, like '00:1a:2b' or "
                "'00-1a-2b'."
            )
        oui = int(re.sub("[:-]", "", prefix), 16)
        return self._prefix(f"{oui:024b}")


mac = MAC(48, "mac")
