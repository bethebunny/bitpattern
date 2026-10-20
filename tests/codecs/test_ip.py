"""IP codecs, checked against the ipaddress module."""

import ipaddress
import random

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bitpattern.codecs import ipv4, ipv6

BLOCKS = st.sampled_from(
    [
        "10.0.0.0/8",
        "10.0.0.0/12",
        "10.1.0.0/16",
        "10.1.2.0/24",
        "10.128.0.0/9",
        "192.168.0.0/16",
        "192.168.1.0/24",
        "0.0.0.0/1",
        "172.16.0.0/12",
    ]
)


def test_cidr():
    assert ipv4.cidr("10.0.0.0/8").size == 2**24
    assert ipv4.cidr("192.168.1.0/24").size == 256
    assert ipv4.cidr("0.0.0.0/0").size == 2**32
    assert ipv6.cidr("2001:db8::/32").size == 2**96


def test_cidr_members():
    assert [str(a) for a in ipv4.cidr("192.168.1.0/30")] == [
        "192.168.1.0",
        "192.168.1.1",
        "192.168.1.2",
        "192.168.1.3",
    ]


def test_cidr_takes_networks():
    assert ipv4.cidr(ipaddress.IPv4Network("10.0.0.0/8")) == ipv4.cidr("10.0.0.0/8")


def test_cidr_refuses_the_other_family():
    with pytest.raises(ValueError):
        ipv4.cidr("2001:db8::/32")


@pytest.mark.parametrize("codec, text", [(ipv4, "::1"), (ipv6, "1.2.3.4")])
def test_encode_refuses_the_other_family(codec, text):
    with pytest.raises(ValueError):
        codec.encode(text)


def test_ipv6_stays_ipv6_below_2_to_the_32():
    """`ipaddress.ip_address` guesses the family, and would guess IPv4."""
    assert ipv6.decode(1) == ipaddress.IPv6Address("::1")
    assert all(a.version == 6 for a in ipv6.cidr("::/126"))
    assert list(ipv6.networks(ipv6.cidr("::/32"))) == [ipaddress.IPv6Network("::/32")]
    assert ipv6.cidr("::/120").choice().version == 6


def test_networks_inverts_cidr():
    for text in ("10.0.0.0/8", "0.0.0.0/0", "192.168.1.1/32", "172.16.0.0/12"):
        block = ipv4.cidr(text)
        assert [str(n) for n in ipv4.networks(block)] == [text]


def test_networks_readme_example():
    spare = ipv4.networks(ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16"))
    assert [str(n) for n in spare][:2] == ["10.0.0.0/16", "10.2.0.0/15"]


@pytest.mark.parametrize(
    "outer, inner",
    [
        ("10.0.0.0/8", "10.1.0.0/16"),
        ("0.0.0.0/0", "127.0.0.0/8"),
        ("192.168.0.0/16", "192.168.1.128/25"),
        ("172.16.0.0/12", "172.16.0.0/12"),
    ],
)
def test_networks_match_address_exclude(outer, inner):
    """`ipaddress` works out the same blocks, independently."""
    a, b = ipaddress.IPv4Network(outer), ipaddress.IPv4Network(inner)
    got = list(ipv4.networks(ipv4.cidr(outer) - ipv4.cidr(inner)))
    assert got == sorted(a.address_exclude(b))


@given(
    chosen=st.lists(BLOCKS, min_size=1, max_size=4), drop=st.lists(BLOCKS, max_size=3)
)
@settings(max_examples=150, deadline=None)
def test_networks_round_trip_through_any_algebra(chosen, drop):
    source = ipv4.none
    for text in chosen:
        source = source | ipv4.cidr(text)
    for text in drop:
        source = source - ipv4.cidr(text)
    networks = list(ipv4.networks(source))
    rebuilt = ipv4.none
    for network in networks:
        rebuilt = rebuilt | ipv4.cidr(network)
    assert rebuilt == source
    assert networks == sorted(set(networks))  # no duplicates, in order
    assert sum(n.num_addresses for n in networks) == source.size


def test_networks_of_a_few_addresses():
    few = ipv4.range("0.0.0.1", "0.0.0.4")
    assert [str(n) for n in ipv4.networks(few)] == ["0.0.0.1/32", "0.0.0.2/31"]


def test_networks_of_the_empty_set():
    assert list(ipv4.networks(ipv4.none)) == []


@pytest.mark.parametrize("source", [ipv6.cidr("::/120")])
def test_networks_refuses_other_codecs(source):
    with pytest.raises(TypeError):
        list(ipv4.networks(source))


@given(chosen=st.lists(BLOCKS, min_size=2, max_size=3))
@settings(max_examples=100, deadline=None)
def test_subset_matches_ipaddress(chosen):
    a, b = chosen[0], chosen[1]
    assume(a != b)
    networks = ipaddress.IPv4Network(a), ipaddress.IPv4Network(b)
    assert (ipv4.cidr(a) <= ipv4.cidr(b)) == networks[0].subnet_of(networks[1])


def test_ip_ranges_can_leave_an_end_open():
    assert ipv4.range() == ipv4.all
    assert [str(a) for a in ipv4.range("255.255.255.254")] == [
        "255.255.255.254",
        "255.255.255.255",
    ]
    assert ipv4.range(stop="0.0.1.0") == ipv4.cidr("0.0.0.0/24")
    assert (str(ipv6.min), str(ipv6.max)) == ("::", "ffff:" * 7 + "ffff")


def test_ip_range():
    assert [str(a) for a in ipv4.range("10.0.0.5", "10.0.0.9")] == [
        "10.0.0.5",
        "10.0.0.6",
        "10.0.0.7",
        "10.0.0.8",
    ]


def test_ip_choice_is_always_a_member():
    block = ipv4.cidr("10.0.0.0/8")
    rng = random.Random(0)
    for _ in range(100):
        assert block.choice(rng=rng) in block


def test_set_algebra_across_blocks():
    private = ipv4.cidr("10.0.0.0/8") | ipv4.cidr("192.168.0.0/16")
    assert private.size == 2**24 + 2**16
    assert ipv4.cidr("10.0.0.0/8") <= private
    assert ipv4.cidr("10.0.0.0/8").isdisjoint(ipv4.cidr("192.168.0.0/16"))
    assert (private & ipv4.cidr("10.1.0.0/16")).size == 2**16


def test_ipv6_sets_index_and_iterate_in_their_top_half():
    upper = ipv6.cidr("8000::/1")
    address = ipaddress.IPv6Address(2**127 + 1)
    assert upper[1] == address and upper.index(address) == 1
    assert upper[-1] == ipaddress.IPv6Address(2**128 - 1)
    tail = ipv6.range("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffd")
    assert [int(address) for address in tail] == [2**128 - 3, 2**128 - 2, 2**128 - 1]
    assert repr(ipv6.all).startswith("BDDSet(ipv6, [IPv6Address('::'), ")


def test_addresses_encode_as_their_ints():
    scoped = ipaddress.IPv6Address("fe80::1%eth0")
    assert ipv6.encode(scoped) == ipv6.encode("fe80::1") == int(scoped)
    assert ipv4.encode(ipaddress.IPv4Address("10.1.2.3")) == 0x0A010203


@pytest.mark.parametrize(
    "codec, interface",
    [
        (ipv4, ipaddress.IPv4Interface("10.1.2.3/8")),
        (ipv6, ipaddress.IPv6Interface("2001:db8::1/64")),
    ],
)
def test_interfaces_arent_addresses(codec, interface):
    """ipaddress doesn't count an interface equal to its address."""
    with pytest.raises(ValueError):
        codec.encode(interface)
    assert interface not in codec.all
