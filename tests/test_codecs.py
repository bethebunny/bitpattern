"""Codecs, with float16 brute-forced over all 65536 of its bit patterns."""

import ipaddress
import itertools
import math
import random

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bitpattern import IntSet
from bitpattern.codecs import Float, float16, float32, float64, ipv4, ipv6

UNIVERSE = range(1 << float16.width)
DECODED = [float16.decode(bits) for bits in UNIVERSE]
SMALLEST_NORMAL = 2.0**-14  # float16 has a 5-bit exponent, with a bias of 15

BOUNDS = (
    -math.inf,
    -65504.0,
    -1.0,
    -SMALLEST_NORMAL,
    -5.960464477539063e-08,
    -0.0,
    0.0,
    5.960464477539063e-08,
    SMALLEST_NORMAL,
    1.0,
    2.0,
    65504.0,
    math.inf,
)

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


def patterns(predicate) -> set[int]:
    """Every float16 bit pattern whose value satisfies `predicate`."""
    return {bits for bits, value in zip(UNIVERSE, DECODED) if predicate(value)}


def brute_force_range(low, high) -> set[int]:
    """Every float16 bit pattern in `[low, high)`, by checking each one."""
    start, stop = float16._key(float16.encode(low)), float16._key(float16.encode(high))
    return {bits for bits in UNIVERSE if start <= float16._key(bits) < stop}


@pytest.mark.parametrize(
    "name, predicate",
    [
        ("nan", math.isnan),
        ("infinities", math.isinf),
        ("finite", math.isfinite),
        ("zeros", lambda v: v == 0),
        ("positive", lambda v: v > 0),
        ("negative", lambda v: v < 0),
        ("nonnegative", lambda v: v >= 0),
        ("sign_clear", lambda v: math.copysign(1, v) > 0),
        ("sign_set", lambda v: math.copysign(1, v) < 0),
        ("subnormal", lambda v: v != 0 and abs(v) < SMALLEST_NORMAL),
    ],
)
def test_named_sets_match_the_decoded_values(name, predicate):
    assert set(getattr(float16, name)) == patterns(predicate)


def test_sign_halves_partition_everything():
    assert set(float16.sign_clear | float16.sign_set) == set(UNIVERSE)
    assert float16.sign_clear.isdisjoint(float16.sign_set)
    assert set(float16.finite | float16.nan | float16.infinities) == set(UNIVERSE)


def test_nan_isnt_positive_or_negative():
    """NaNs have a sign bit, but no order."""
    assert float16.nan.isdisjoint(float16.positive)
    assert float16.nan.isdisjoint(float16.negative)
    assert not float16.nan.isdisjoint(float16.sign_clear)
    assert not float16.nan.isdisjoint(float16.sign_set)


@pytest.mark.parametrize("codec", [float16, float32, float64])
def test_named_set_sizes(codec):
    half, mantissa = 1 << (codec.width - 1), 1 << codec.mantissa
    assert codec.all.size == 1 << codec.width
    assert codec.sign_clear.size == half == codec.sign_set.size
    assert codec.zeros.size == 2
    assert codec.infinities.size == 2
    assert codec.nan.size == 2 * mantissa - 2
    assert codec.subnormal.size == 2 * mantissa - 2
    assert codec.positive.size == half - mantissa == codec.negative.size
    assert codec.nonnegative.size == half - mantissa + 2
    assert codec.finite.size == (1 << codec.width) - 2 * mantissa


def test_float64_sizes():
    assert float64.nan.size == 2**53 - 2
    assert (float64.all - float64.nan - float64.infinities).size == 2**64 - 2**53
    assert float64.positive.size == 2**63 - 2**52


def test_named_sets_are_cached():
    assert float64.nan is float64.nan


def test_key_sorts_bits_by_value():
    ordered = sorted(patterns(lambda v: not math.isnan(v)), key=float16._key)
    values = [float16.decode(bits) for bits in ordered]
    assert all(a <= b for a, b in itertools.pairwise(values))
    assert values[0] == -math.inf and values[-1] == math.inf
    # -0.0 sorts before 0.0, even though they're ==
    assert math.copysign(1, values[len(values) // 2 - 1]) < 0
    assert math.copysign(1, values[len(values) // 2]) > 0


def test_non_negative_bits_count_up_in_value():
    """That's what makes index 0 the natural thing to shrink towards."""
    values = list(float16.values(float16.finite & float16.sign_clear))
    assert values[0] == 0.0 and math.copysign(1, values[0]) > 0
    assert values[-1] == 65504.0
    assert values == sorted(values)


def test_minus_zero_has_the_highest_bits():
    """`-0.0` is in `nonnegative`, but its sign bit is set."""
    values = list(float16.values(float16.finite & float16.nonnegative))
    assert math.copysign(1, values[-1]) < 0 and values[-1] == 0.0


@pytest.mark.parametrize("low", BOUNDS)
def test_float_range_agrees_with_brute_force(low):
    for high in BOUNDS:
        assert set(float16.range(low, high)) == brute_force_range(low, high)


def test_float_range_is_half_open():
    assert float16.encode(1.0) in float16.range(1.0, 2.0)
    assert float16.encode(2.0) not in float16.range(1.0, 2.0)
    assert float16.range(1.0, 1.0).size == 0
    assert float16.range(2.0, 1.0).size == 0


def test_float_range_never_includes_nan():
    spread = float16.range(-math.inf, math.inf)
    assert spread.isdisjoint(float16.nan)
    assert float16.encode(-math.inf) in spread
    assert float16.encode(math.inf) not in spread


def test_nan_cant_bound_a_range():
    with pytest.raises(ValueError):
        float16.range(math.nan, 1.0)
    with pytest.raises(ValueError):
        float16.range(0.0, math.nan)


def test_wide_float_ranges_stay_small():
    """A range is one node per bit, however many floats are in it."""
    wide = float64.range(-1e300, 1e300)
    assert wide.size > 2**63
    assert len(str(wide.bdd)) < 10_000


def test_float64_range_ends():
    assert float64.decode(float64.range(1.0, 2.0)[0]) == 1.0
    unit = float64.range(0.0, 1.0)
    assert float64.decode(unit[0]) == 0.0
    assert float64.decode(unit[-1]) == math.nextafter(1.0, 0.0)


@pytest.mark.parametrize("codec", [float16, float32, float64])
def test_floats_round_trip(codec):
    for value in (0.0, 1.0, -1.0, 2.0, 0.5, math.inf, -math.inf):
        assert codec.decode(codec.encode(value)) == value
    assert math.copysign(1, codec.decode(codec.encode(-0.0))) < 0


@given(value=st.floats(allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_float64_round_trips_any_value(value):
    assert float64.decode(float64.encode(value)) == value


def test_float_choice_is_always_a_member():
    source = float16.finite & float16.positive
    rng = random.Random(0)
    for _ in range(200):
        value = float16.choice(source, rng)
        assert math.isfinite(value) and value > 0
        assert float16.encode(value) in source


def test_float_choice_reaches_the_whole_set():
    tiny = float16.range(1.0, 1.001)
    seen = {float16.choice(tiny, random.Random(seed)) for seed in range(60)}
    assert seen == set(float16.values(tiny))


def test_pattern_has_to_be_the_codecs_width():
    assert float16.pattern("?111.11??.????.????").size == float16.nan.size + 2
    with pytest.raises(ValueError):
        float16.pattern("0000")


def test_set_and_values_round_trip():
    source = float16.set([1.0, -2.0, 0.5])
    assert sorted(float16.values(source)) == [-2.0, 0.5, 1.0]


def test_a_custom_float_format():
    assert Float(width=16, name="float16", exponent=5, format=">e") == float16


def test_cidr():
    assert ipv4.cidr("10.0.0.0/8").size == 2**24
    assert ipv4.cidr("192.168.1.0/24").size == 256
    assert ipv4.cidr("0.0.0.0/0").size == 2**32
    assert ipv6.cidr("2001:db8::/32").size == 2**96


def test_cidr_members():
    block = ipv4.cidr("192.168.1.0/30")
    assert [str(a) for a in ipv4.values(block)] == [
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
    assert all(a.version == 6 for a in ipv6.values(ipv6.cidr("::/126")))
    assert list(ipv6.networks(ipv6.cidr("::/32"))) == [ipaddress.IPv6Network("::/32")]
    assert ipv6.choice(ipv6.cidr("::/120")).version == 6


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


def test_networks_of_a_narrower_set():
    assert [str(n) for n in ipv4.networks(IntSet([1, 2, 3], 2))] == [
        "0.0.0.1/32",
        "0.0.0.2/31",
    ]


def test_networks_of_a_wider_set_whose_members_fit():
    assert [str(n) for n in ipv4.networks(IntSet([1], 40))] == ["0.0.0.1/32"]
    assert [str(n) for n in ipv4.networks(IntSet.range(0, 5, 40))] == [
        "0.0.0.0/30",
        "0.0.0.4/32",
    ]


def test_networks_refuses_members_that_dont_fit():
    with pytest.raises(ValueError, match="wider than 32 bits"):
        list(ipv4.networks(IntSet([1 << 35])))


def test_networks_of_the_empty_set():
    assert list(ipv4.networks(ipv4.none)) == []
    assert list(ipv4.networks(IntSet([], 40))) == []


@given(chosen=st.lists(BLOCKS, min_size=2, max_size=3))
@settings(max_examples=100, deadline=None)
def test_subset_matches_ipaddress(chosen):
    a, b = chosen[0], chosen[1]
    assume(a != b)
    networks = ipaddress.IPv4Network(a), ipaddress.IPv4Network(b)
    assert (ipv4.cidr(a) <= ipv4.cidr(b)) == networks[0].subnet_of(networks[1])


def test_ip_range():
    span = ipv4.range("10.0.0.5", "10.0.0.9")
    assert [str(a) for a in ipv4.values(span)] == [
        "10.0.0.5",
        "10.0.0.6",
        "10.0.0.7",
        "10.0.0.8",
    ]


def test_ip_choice_is_always_a_member():
    block = ipv4.cidr("10.0.0.0/8")
    rng = random.Random(0)
    for _ in range(100):
        assert ipv4.encode(ipv4.choice(block, rng)) in block


def test_set_algebra_across_blocks():
    private = ipv4.cidr("10.0.0.0/8") | ipv4.cidr("192.168.0.0/16")
    assert private.size == 2**24 + 2**16
    assert ipv4.cidr("10.0.0.0/8") <= private
    assert ipv4.cidr("10.0.0.0/8").isdisjoint(ipv4.cidr("192.168.0.0/16"))
    assert (private & ipv4.cidr("10.1.0.0/16")).size == 2**16


@pytest.mark.parametrize("codec", [float16, float32, float64, ipv4, ipv6])
def test_all_and_none(codec):
    assert codec.all.size == 1 << codec.width
    assert codec.none.size == 0
    assert codec.all.width == codec.width == codec.none.width
    assert ~codec.all == codec.none


def test_choice_defaults_to_everything():
    assert isinstance(float16.choice(rng=random.Random(0)), float)
    assert ipv4.choice(rng=random.Random(0)) in ipaddress.IPv4Network("0.0.0.0/0")


def test_choice_from_an_empty_set_raises():
    with pytest.raises(IndexError):
        float16.choice(float16.none)


def test_codec_sets_are_plain_intsets():
    assert type(float64.nan) is IntSet
    assert type(ipv4.cidr("10.0.0.0/8")) is IntSet
