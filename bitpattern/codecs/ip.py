"""IPv4 and IPv6 addresses as their integers."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..bdd import blocks
from ..intset import IntSet
from .codec import BDDSet, OrderedCodec

__all__ = ["IP", "ipv4", "ipv6"]

Address = TypeVar("Address", ipaddress.IPv4Address, ipaddress.IPv6Address)
Network = TypeVar("Network", ipaddress.IPv4Network, ipaddress.IPv6Network)


@dataclass(frozen=True, repr=False)
class IP(OrderedCodec[Address], Generic[Address, Network]):
    """IPv4 or IPv6 addresses."""

    # The family's own classes. ipaddress.ip_address() would decode IPv6 addresses
    # below 2**32 as IPv4.
    address_type: type[Address]
    network_type: type[Network]

    def encode(self, value: object) -> int:
        """Accepts addresses, strings or ints."""
        if isinstance(value, self.address_type) and type(value) is self.address_type:
            return int(value)
        if not isinstance(value, (self.address_type, str, int)):
            raise TypeError(f"{self} can't encode {value!r}")
        # A subclass goes through its text, so an interface (with a prefix) is refused.
        return int(self.address_type(value))  # raises ValueError if malformed

    def decode(self, bits: int) -> Address:
        try:
            return self.address_type(bits)
        except ValueError as e:  # wider than this family's addresses
            raise IndexError(f"{self} has no value encoded as {bits}") from e

    def cidr(self, network: Network | str) -> BDDSet[Address]:
        block = self.network_type(network, strict=False)
        start = int(block.network_address)
        return BDDSet(
            self, IntSet.range(start, start + block.num_addresses, width=self.width)
        )

    def range(
        self,
        start: Address | str | int | None = None,
        stop: Address | str | int | None = None,
    ) -> BDDSet[Address]:
        """Addresses in `[start, stop)`. `None` is unbounded."""
        address = self.address_type
        return super().range(
            None if start is None else address(self.encode(start)),
            None if stop is None else address(self.encode(stop)),
        )

    def networks(self, source: BDDSet[Address]) -> Iterator[Network]:
        """The fewest CIDR blocks that exactly cover `source`, in order."""
        if source.codec != self:
            raise TypeError(f"{source.codec} addresses aren't {self} addresses")
        for start, free in blocks(source.storage.bdd, width=self.width):
            yield self.network_type((start, self.width - free))


ipv4 = IP(32, "ipv4", ipaddress.IPv4Address, ipaddress.IPv4Network)
ipv6 = IP(128, "ipv6", ipaddress.IPv6Address, ipaddress.IPv6Network)
