"""Codecs and BDDSets, with float16 brute-forced over all 65536 of its bit patterns."""

import ipaddress
import itertools
import math
import operator
import random
from functools import cached_property

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bitpattern import IntSet, Pattern
from bitpattern.codecs import (
    BDDSet,
    Codec,
    Float,
    float16,
    float32,
    float64,
    ipv4,
    ipv6,
)

UNIVERSE = range(1 << float16.width)
DECODED = [float16.decode(bits) for bits in UNIVERSE]
SMALLEST_NORMAL = 2.0**-14  # float16 has a 5-bit exponent, with a bias of 15
SMALL = float16.range(1.0, 1.01)  # the 10 float16s from 1.0

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
    assert set(getattr(float16, name).storage) == patterns(predicate)


def test_sign_halves_partition_everything():
    assert set((float16.sign_clear | float16.sign_set).storage) == set(UNIVERSE)
    assert float16.sign_clear.isdisjoint(float16.sign_set)
    everything = float16.finite | float16.nan | float16.infinities
    assert set(everything.storage) == set(UNIVERSE)


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
    values = list(float16.finite & float16.sign_clear)
    assert values[0] == 0.0 and math.copysign(1, values[0]) > 0
    assert values[-1] == 65504.0
    assert values == sorted(values)


def test_minus_zero_has_the_highest_bits():
    """`-0.0` is in `nonnegative`, but its sign bit is set."""
    values = list(float16.finite & float16.nonnegative)
    assert math.copysign(1, values[-1]) < 0 and values[-1] == 0.0


@pytest.mark.parametrize("low", BOUNDS)
def test_float_range_agrees_with_brute_force(low):
    for high in BOUNDS:
        assert set(float16.range(low, high).storage) == brute_force_range(low, high)


def test_float_range_is_half_open():
    assert 1.0 in float16.range(1.0, 2.0)
    assert 2.0 not in float16.range(1.0, 2.0)
    assert float16.range(1.0, 1.0).size == 0
    assert float16.range(2.0, 1.0).size == 0


def test_float_range_never_includes_nan():
    spread = float16.range(-math.inf, math.inf)
    assert spread.isdisjoint(float16.nan)
    assert -math.inf in spread
    assert math.inf not in spread


def test_nan_cant_bound_a_range():
    with pytest.raises(ValueError):
        float16.range(math.nan, 1.0)
    with pytest.raises(ValueError):
        float16.range(0.0, math.nan)


def test_wide_float_ranges_stay_small():
    """A range is one node per bit, however many floats are in it."""
    wide = float64.range(-1e300, 1e300)
    assert wide.size > 2**63
    assert len(str(wide.storage.bdd)) < 10_000


def test_float64_range_ends():
    assert float64.range(1.0, 2.0)[0] == 1.0
    unit = float64.range(0.0, 1.0)
    assert unit[0] == 0.0
    assert unit[-1] == math.nextafter(1.0, 0.0)


@pytest.mark.parametrize("codec", [float16, float32, float64])
def test_floats_round_trip(codec):
    for value in (0.0, 1.0, -1.0, 2.0, 0.5, math.inf, -math.inf):
        assert codec.decode(codec.encode(value)) == value
    assert math.copysign(1, codec.decode(codec.encode(-0.0))) < 0


@given(value=st.floats(allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_float64_round_trips_any_value(value):
    assert float64.decode(float64.encode(value)) == value


@pytest.mark.parametrize("value", ["x", None, 1e10])  # 1e10 is too big for a float16
def test_encode_refuses_what_it_cant_encode(value):
    with pytest.raises(ValueError):
        float16.encode(value)


def test_float_choice_is_always_a_member():
    source = float16.finite & float16.positive
    rng = random.Random(0)
    for _ in range(200):
        value = source.choice(rng=rng)
        assert math.isfinite(value) and value > 0
        assert value in source


def test_float_choice_reaches_the_whole_set():
    seen = {SMALL.choice(rng=random.Random(seed)) for seed in range(100)}
    assert seen == set(SMALL)


def test_pattern_has_to_be_the_codecs_width():
    assert float16.pattern("?111.11??.????.????").size == float16.nan.size + 2
    with pytest.raises(ValueError):
        float16.pattern("0000")


def test_set_round_trips():
    assert sorted(float16.set([1.0, -2.0, 0.5])) == [-2.0, 0.5, 1.0]


def test_a_custom_float_format():
    assert Float(width=16, name="float16", exponent=5, format=">e") == float16


def test_bdd_sets_hold_values():
    supported = float64.finite - float64.subnormal
    assert 0.25 in supported and -0.0 in supported
    assert math.inf not in supported
    assert math.inf - math.inf in float64.nan and 0.25 not in float64.nan
    assert "10.1.2.3" in ipv4.cidr("10.0.0.0/8")


@pytest.mark.parametrize("value", ["x", None, 1e10, [1.0]])
def test_values_a_codec_cant_encode_arent_members(value):
    assert value not in float16.all
    assert value not in ipv4.all


def test_bdd_sets_read_like_lists_of_their_values():
    values = list(SMALL)
    assert len(values) == len(SMALL) == SMALL.size == 10
    assert values == sorted(values) and all(1.0 <= v < 1.01 for v in values)
    assert [SMALL[at] for at in range(10)] == values
    assert SMALL[-1] == values[-1]
    assert list(reversed(SMALL)) == values[::-1]
    assert [SMALL.index(v) for v in values] == list(range(10))
    assert SMALL.count(values[3]) == 1 and SMALL.count(2.0) == 0
    with pytest.raises(ValueError):
        SMALL.index(2.0)


def test_slices_are_bdd_sets():
    assert type(SMALL[2:5]) is BDDSet
    assert list(SMALL[2:5]) == list(SMALL)[2:5]


def test_bdd_sets_combine_with_the_same_codec():
    a, b = float16.range(1.0, 2.0), float16.range(1.5, 3.0)
    assert a & b == float16.range(1.5, 2.0)
    assert a | b == float16.range(1.0, 3.0)
    assert a - b == float16.range(1.0, 1.5)
    assert a ^ b == (a - b) | (b - a)
    assert ~float16.all == float16.none
    assert float16.range(1.5, 2.0) <= a and not a <= b
    assert a.isdisjoint(float16.range(2.0, 3.0))


class Digit(Codec[int]):
    """0 to 9, which leaves 6 of its 16 bit patterns unused."""

    def encode(self, value: object) -> int:
        if isinstance(value, int) and 0 <= value <= 9:
            return value
        raise ValueError(f"{value!r} isn't a digit")

    def decode(self, bits: int) -> int:
        return bits

    @cached_property
    def all(self) -> BDDSet[int]:
        return self.set(range(10))


def test_invert_stays_inside_the_codecs_values():
    digit = Digit(4, "digit")
    assert ~digit.set([0, 1]) == digit.set(range(2, 10))
    assert ~digit.none == digit.all


@pytest.mark.parametrize(
    "other", [float32.range(1.0, 2.0), IntSet([1], width=16), {1.5}, [1.5], 5]
)
def test_bdd_sets_dont_combine_with_anything_else(other):
    """Not even a plain IntSet, which would be ambiguous between bits and values."""
    for operate in (operator.and_, operator.or_, operator.sub, operator.le):
        with pytest.raises(TypeError):
            operate(SMALL, other)


def test_bdd_sets_need_the_codecs_width():
    with pytest.raises(ValueError):
        BDDSet(ipv4, IntSet([1], width=8))


def test_bdd_sets_compare_and_hash_by_codec_and_bits():
    one = float16.range(1.0, 1.0009765625)  # the next float16 after 1.0
    assert one == float16.set([1.0]) and hash(one) == hash(float16.set([1.0]))
    assert float16.set([1.0]) != float32.set([1.0])
    assert float16.set([1.0]) != float16.set([1.0]).storage


def test_bdd_set_reprs():
    assert repr(float16.set([1.0, 2.0])) == "BDDSet(float16, [1.0, 2.0])"
    assert repr(float64.finite).startswith(
        "BDDSet(float64, [0.0, 5e-324, 1e-323, 1.5e-323, ..., "
    )
    assert "size=2**24" in repr(ipv4.cidr("10.0.0.0/8"))


def test_big_bdd_sets_dont_go_through_len():
    assert float64.finite
    assert float64.finite.size == 2**64 - 2**53
    assert float64.finite[-1] == -1.7976931348623157e308
    assert next(reversed(float64.finite)) == -1.7976931348623157e308


def test_codec_sets_are_bdd_sets():
    assert type(float64.nan) is BDDSet and type(float64.nan.storage) is IntSet
    assert type(float64.pattern("*" + ".*" * 15).storage) is Pattern


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


@pytest.mark.parametrize("codec", [float16, float32, float64, ipv4, ipv6])
def test_all_and_none(codec):
    assert codec.all.size == 1 << codec.width
    assert codec.none.size == 0
    assert ~codec.all == codec.none


def test_choice_defaults_to_the_random_module():
    assert isinstance(float16.all.choice(), float)
    assert ipv4.all.choice(rng=random.Random(0)) in ipaddress.IPv4Network("0.0.0.0/0")


def test_choice_from_an_empty_set_raises():
    with pytest.raises(IndexError):
        float16.none.choice()
