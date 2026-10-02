"""IPv4 and IPv6 addresses, which are integers already."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..bdd import BDD, cofactors
from ..intset import IntSet
from .codec import BDDSet, Codec

__all__ = ["IP", "ipv4", "ipv6"]

Address = TypeVar("Address", ipaddress.IPv4Address, ipaddress.IPv6Address)
Network = TypeVar("Network", ipaddress.IPv4Network, ipaddress.IPv6Network)


@dataclass(frozen=True, repr=False)
class IP(Codec[Address], Generic[Address, Network]):
    """IPv4 or IPv6 addresses."""

    # The family's own classes, since ipaddress.ip_address() guesses the family
    # from the value, and would decode every IPv6 address below 2**32 as IPv4.
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
        self, low: Address | str | int, high: Address | str | int
    ) -> BDDSet[Address]:
        """Addresses in `[low, high)`."""
        return BDDSet(
            self, IntSet.range(self.encode(low), self.encode(high), width=self.width)
        )

    def networks(self, source: BDDSet[Address]) -> Iterator[Network]:
        """The fewest CIDR blocks that make up `source`, ie. route aggregation.

        Each path to ACCEPT is a block, and since the diagram is reduced, each one
        is as big as it can be. Unlike a pattern, a block's free bits all have to
        be at the end, so this splits even the bits that the diagram doesn't test.
        """

        def walk(node: BDD, bit: int, prefix: int) -> Iterator[Network]:
            if node is BDD.ACCEPT:
                yield self.network_type((prefix << (bit + 1), self.width - 1 - bit))
            elif node:
                low, high = cofactors(node, bit)
                yield from walk(low, bit - 1, prefix << 1)
                yield from walk(high, bit - 1, prefix << 1 | 1)

        if source.codec != self:
            raise TypeError(f"{source.codec} addresses aren't {self} addresses")
        yield from walk(source.bits.bdd, self.width - 1, 0)


ipv4 = IP(32, "ipv4", ipaddress.IPv4Address, ipaddress.IPv4Network)
ipv6 = IP(128, "ipv6", ipaddress.IPv6Address, ipaddress.IPv6Network)
