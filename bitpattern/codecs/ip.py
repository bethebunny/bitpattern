"""IPv4 and IPv6 addresses as their integers."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..bdd import BDD, cofactors
from ..intset import IntSet
from .codec import BDDSet, OrderedCodec

__all__ = ["IP", "ipv4", "ipv6"]

Address = TypeVar("Address", ipaddress.IPv4Address, ipaddress.IPv6Address)
Network = TypeVar("Network", ipaddress.IPv4Network, ipaddress.IPv6Network)


@dataclass(frozen=True, repr=False)
class IP(OrderedCodec[Address], Generic[Address, Network]):
    """IPv4 or IPv6 addresses.

    `encode` accepts addresses, their strings or their integers.
    """

    # The family's own classes. ipaddress.ip_address() would decode IPv6 addresses
    # below 2**32 as IPv4.
    address_type: type[Address]
    network_type: type[Network]

    def encode(self, value: object) -> int:
        return int(self.address_type(value))

    def decode(self, bits: int) -> Address:
        return self.address_type(bits)

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
        """Addresses in `[start, stop)`. A bound of `None` leaves that end open."""
        address = self.address_type
        return super().range(
            None if start is None else address(start),
            None if stop is None else address(stop),
        )

    def networks(self, source: BDDSet[Address]) -> Iterator[Network]:
        """The fewest CIDR blocks that make up `source`, in order, ie. route
        aggregation."""

        # Each path to ACCEPT is a block, as big as it can be in a reduced diagram.
        # Blocks only have free bits at the end, so this splits bits the path skips.
        def walk(node: BDD, bit: int, prefix: int) -> Iterator[Network]:
            if node is BDD.ACCEPT:
                yield self.network_type((prefix << (bit + 1), self.width - 1 - bit))
            elif node:
                low, high = cofactors(node, bit)
                yield from walk(low, bit - 1, prefix << 1)
                yield from walk(high, bit - 1, prefix << 1 | 1)

        if source.codec != self:
            raise TypeError(f"{source.codec} addresses aren't {self} addresses")
        yield from walk(source.storage.bdd, self.width - 1, 0)


ipv4 = IP(32, "ipv4", ipaddress.IPv4Address, ipaddress.IPv4Network)
ipv6 = IP(128, "ipv6", ipaddress.IPv6Address, ipaddress.IPv6Network)
