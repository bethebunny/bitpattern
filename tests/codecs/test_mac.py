"""MAC address codec, with the flag sets brute-forced over every first octet."""

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern import IntSet
from bitpattern.codecs import BDDSet
from bitpattern.codecs.mac import mac

FIRST_OCTETS = range(256)

ADDRESS_BITS = st.integers(0, 2**48 - 1)


def first_octet(octet: int) -> BDDSet[str]:
    """The 2**40 addresses whose first octet is `octet`."""
    return BDDSet(mac, IntSet.range(octet << 40, (octet + 1) << 40, width=48))


def spellings(bits: int) -> list[str]:
    """The address `bits` with colons, hyphens and Cisco's dots, lowercase first."""
    digits = f"{bits:012x}"
    octets, groups = re.findall("..", digits), re.findall("....", digits)
    lower = [":".join(octets), "-".join(octets), ".".join(groups)]
    return lower + [text.upper() for text in lower]


@pytest.mark.parametrize(
    "name, predicate",
    [
        ("multicast", lambda octet: octet & 1),
        ("unicast", lambda octet: not octet & 1),
        ("local", lambda octet: octet & 2),
        ("universal", lambda octet: not octet & 2),
    ],
)
def test_flag_sets_match_the_first_octet(name, predicate):
    flagged = getattr(mac, name)
    for octet in FIRST_OCTETS:
        block = first_octet(octet)
        assert flagged & block == (block if predicate(octet) else mac.none)
        assert (f"{octet:02x}:23:45:67:89:ab" in flagged) == bool(predicate(octet))


def test_broadcast_is_every_bit_set():
    assert list(mac.broadcast) == ["ff:ff:ff:ff:ff:ff"]
    assert mac.broadcast == mac.all[-1:]
    assert mac.broadcast <= mac.multicast & mac.local


def test_patterns_match_the_octets_in_order():
    assert mac.pattern("*.???1" + ".*" * 10) == mac.multicast
    assert mac.pattern("*.??1?" + ".*" * 10) == mac.local
    iana = mac.pattern("0000.0001.0000.0000.0101.1110" + ".*" * 6)
    assert iana == mac.oui("01:00:5e")


def test_macs_encode_with_the_first_octet_highest():
    for text in ("01:23:45:67:89:ab", "01-23-45-67-89-AB", "0123.4567.89aB"):
        assert mac.encode(text) == 0x0123456789AB
    assert mac.decode(0x0123456789AB) == "01:23:45:67:89:ab"
    assert mac.all[0] == "00:00:00:00:00:00"


@given(bits=ADDRESS_BITS)
@settings(deadline=None)
def test_every_spelling_encodes_to_the_same_bits(bits):
    assert [mac.encode(text) for text in spellings(bits)] == [bits] * 6
    assert mac.decode(bits) == spellings(bits)[0]


@given(encodings=st.lists(ADDRESS_BITS))
@settings(deadline=None)
def test_macs_sort_like_their_strings(encodings):
    addresses = [mac.decode(bits) for bits in encodings]
    upper = [address.upper() for address in addresses]
    assert list(mac.set(addresses + upper)) == sorted(set(addresses))


@pytest.mark.parametrize(
    "value",
    [
        "0123456789ab",
        "01:23:45:67:89",
        "01:23:45:67:89:ab:cd",
        "1:23:45:67:89:ab",  # octets need both digits
        "01:23-45:67:89:ab",  # one kind of separator throughout
        "0123:4567:89ab",  # groups of four only take dots
        "01.23.45.67.89.ab",
        "01:23:45:67:89:ag",
        " 01:23:45:67:89:ab",
        "01:23:45:67:89:ab\n",
        "０１:23:45:67:89:ab",  # fullwidth digits, which int() reads
        "",
    ],
)
def test_macs_refuse_anything_else(value):
    with pytest.raises(ValueError):
        mac.encode(value)
    assert value not in mac.all


@pytest.mark.parametrize("value", [0, 0x0123456789AB, True, b"01:23:45:67:89:ab", None])
def test_macs_refuse_what_isnt_a_string(value):
    """An int would be the bits, which is what `decode` takes."""
    with pytest.raises(TypeError):
        mac.encode(value)
    assert value not in mac.all


def test_oui_is_the_block_of_addresses_under_it():
    block = mac.oui("00:1a:2b")
    assert block.size == 2**24
    assert block[0] == "00:1a:2b:00:00:00"
    assert block[-1] == "00:1a:2b:ff:ff:ff"
    assert mac.oui("00-1A-2B") == block


@given(oui=st.integers(0, 2**24 - 1), bits=ADDRESS_BITS)
@settings(deadline=None)
def test_oui_holds_the_addresses_that_start_with_it(oui, bits):
    prefix = ":".join(re.findall("..", f"{oui:06x}"))
    block = mac.oui(prefix)
    assert block == BDDSet(mac, IntSet.range(oui << 24, (oui + 1) << 24, width=48))
    address = mac.decode(bits)
    assert (address in block) == address.startswith(prefix)
    assert prefix + address[8:] in block


def test_ouis_have_the_flags_of_their_first_octet():
    assert mac.oui("01:00:5e") <= mac.multicast  # IANA's, used for IPv4 multicast
    assert mac.oui("00:1a:2b") <= mac.unicast & mac.universal
    assert mac.oui("02:00:00") <= mac.local


@pytest.mark.parametrize(
    "prefix",
    ["00:1a", "00:1a:2b:3c", "001a2b", "00:1a-2b", "0:1a:2b", "00.1a.2b", "00:1a:2g"],
)
def test_oui_refuses_anything_but_three_octets(prefix):
    with pytest.raises(ValueError):
        mac.oui(prefix)


def test_macs_are_ordered_like_their_strings():
    assert (mac.min, mac.max) == ("00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff")
    top = mac.range("ff:ff:ff:ff:ff:fe")
    assert list(top) == ["ff:ff:ff:ff:ff:fe", "ff:ff:ff:ff:ff:ff"]
    assert mac.range(stop="00:00:00:00:01:00").size == 256
    assert mac.range("00:1a:2b:00:00:00", "00:1a:2c:00:00:00") == mac.oui("00:1a:2b")


def test_mac_sets_repr_as_addresses():
    assert repr(mac) == "mac"
    assert repr(mac.broadcast) == "BDDSet(mac, ['ff:ff:ff:ff:ff:ff'])"
    assert repr(mac.oui("00:1a:2b")).endswith("'00:1a:2b:ff:ff:ff'], size=2**24)")
