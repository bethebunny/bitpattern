"""Codecs, with float16 brute-forced over its whole 65536-pattern universe."""

import ipaddress
import math
import random

import pytest
from hypothesis import assume, given, settings, strategies as st

from bitpattern import IntSet
from bitpattern.codecs import Float, float16, float32, float64, ipv4, ipv6

UNIVERSE = range(1 << float16.width)
DECODED = [float16.decode(bits) for bits in UNIVERSE]
SMALLEST_NORMAL = 2.0**-14  # float16 has a 5-bit exponent, bias 15


def patterns(predicate) -> set[int]:
    """Every float16 bit pattern whose decoded value satisfies `predicate`."""
    return {bits for bits, value in zip(UNIVERSE, DECODED) if predicate(value)}


class TestFloatNamedSets:
    """Each named set against a predicate on the decoded value."""

    @pytest.mark.parametrize(
        "name, predicate",
        [
            ("nan", lambda v: v != v),
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
    def test_matches_the_decoded_predicate(self, name, predicate):
        assert set(getattr(float16, name)) == patterns(predicate)

    def test_the_halves_partition_the_universe(self):
        assert set(float16.sign_clear | float16.sign_set) == set(UNIVERSE)
        assert float16.sign_clear.isdisjoint(float16.sign_set)
        assert set(float16.finite | float16.nan | float16.infinities) == set(UNIVERSE)

    def test_a_nan_is_neither_positive_nor_negative(self):
        """The trap the sign-bit reading sets: NaN has a sign but no order."""
        assert float16.nan.isdisjoint(float16.positive)
        assert float16.nan.isdisjoint(float16.negative)
        assert not float16.nan.isdisjoint(float16.sign_clear)
        assert not float16.nan.isdisjoint(float16.sign_set)

    @pytest.mark.parametrize("codec", [float16, float32, float64])
    def test_cardinalities(self, codec):
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

    def test_the_float64_headline_figures(self):
        assert float64.nan.size == 2**53 - 2
        assert (float64.all - float64.nan - float64.infinities).size == 2**64 - 2**53
        assert float64.positive.size == 2**63 - 2**52


class TestFloatOrdering:
    def test_key_orders_patterns_by_value(self):
        """Sorting non-NaN patterns by `key` must sort them by float value."""
        ordered = sorted(patterns(lambda v: v == v), key=float16._key)
        values = [float16.decode(bits) for bits in ordered]
        assert all(a <= b for a, b in zip(values, values[1:]))
        assert values[0] == -math.inf and values[-1] == math.inf
        # The total order separates the zeros, which `==` does not.
        assert math.copysign(1, values[len(values) // 2 - 1]) < 0
        assert math.copysign(1, values[len(values) // 2]) > 0

    def test_sign_clear_patterns_index_in_increasing_magnitude(self):
        """The property that makes index 0 the natural shrink target."""
        values = list(float16.values(float16.finite & float16.sign_clear))
        assert values[0] == 0.0 and math.copysign(1, values[0]) > 0
        assert values[-1] == 65504.0
        assert values == sorted(values)

    def test_minus_zero_sorts_last_in_bit_order(self):
        """`-0.0` is in `nonnegative` by value but has the highest bit pattern."""
        values = list(float16.values(float16.finite & float16.nonnegative))
        assert math.copysign(1, values[-1]) < 0 and values[-1] == 0.0


class TestFloatRange:
    """`range` against brute force over the whole float16 universe."""

    bounds = [
        -math.inf, -65504.0, -1.0, -SMALLEST_NORMAL, -5.960464477539063e-08, -0.0,
        0.0, 5.960464477539063e-08, SMALLEST_NORMAL, 1.0, 2.0, 65504.0, math.inf,
    ]

    def expected(self, low, high):
        start, stop = float16._key(float16.encode(low)), float16._key(float16.encode(high))
        return {bits for bits in UNIVERSE if start <= float16._key(bits) < stop}

    @pytest.mark.parametrize("low", bounds)
    def test_agrees_with_brute_force(self, low):
        for high in self.bounds:
            assert set(float16.range(low, high)) == self.expected(low, high)

    def test_is_half_open(self):
        assert float16.encode(1.0) in float16.range(1.0, 2.0)
        assert float16.encode(2.0) not in float16.range(1.0, 2.0)
        assert float16.range(1.0, 1.0).size == 0
        assert float16.range(2.0, 1.0).size == 0

    def test_finite_bounds_exclude_nan(self):
        spread = float16.range(-math.inf, math.inf)
        assert spread.isdisjoint(float16.nan)
        assert float16.encode(-math.inf) in spread
        assert float16.encode(math.inf) not in spread

    def test_rejects_a_nan_bound(self):
        with pytest.raises(ValueError):
            float16.range(math.nan, 1.0)
        with pytest.raises(ValueError):
            float16.range(0.0, math.nan)

    def test_wide_ranges_stay_small(self):
        """A range is one node per bit, whatever it holds."""
        wide = float64.range(-1e300, 1e300)
        assert wide.size > 2**63
        assert len(str(wide.bdd)) < 10_000  # not enumerated, just a diagram

    def test_float64_range_endpoints(self):
        assert float64.decode(float64.range(1.0, 2.0)[0]) == 1.0
        unit = float64.range(0.0, 1.0)
        assert float64.decode(unit[0]) == 0.0
        assert float64.decode(unit[-1]) == math.nextafter(1.0, 0.0)


class TestFloatCodec:
    @pytest.mark.parametrize("codec", [float16, float32, float64])
    def test_round_trips(self, codec):
        for value in (0.0, 1.0, -1.0, 2.0, 0.5, math.inf, -math.inf):
            assert codec.decode(codec.encode(value)) == value
        assert math.copysign(1, codec.decode(codec.encode(-0.0))) < 0

    @given(value=st.floats(allow_nan=False))
    @settings(max_examples=200, deadline=None)
    def test_float64_round_trips_any_value(self, value):
        assert float64.decode(float64.encode(value)) == value

    def test_sample_is_always_a_member(self):
        source = float16.finite & float16.positive
        rng = random.Random(0)
        for _ in range(200):
            value = float16.sample(source, rng)
            assert math.isfinite(value) and value > 0
            assert float16.encode(value) in source

    def test_sample_reaches_the_whole_set(self):
        tiny = float16.range(1.0, 1.001)
        seen = {float16.sample(tiny, random.Random(seed)) for seed in range(60)}
        assert seen == set(float16.values(tiny))

    def test_pattern_must_match_the_width(self):
        assert float16.pattern('?111.11??.????.????').size == float16.nan.size + 2
        with pytest.raises(ValueError):
            float16.pattern('0000')

    def test_set_and_values_round_trip(self):
        source = float16.set([1.0, -2.0, 0.5])
        assert sorted(float16.values(source)) == [-2.0, 0.5, 1.0]

    def test_codecs_repr_as_their_names(self):
        assert [repr(c) for c in (float16, float32, float64)] == ["float16", "float32", "float64"]

    def test_a_custom_format(self):
        assert Float(width=16, name="float16", exponent=5, format=">e") == float16


class TestIP:
    def test_cidr(self):
        assert ipv4.cidr("10.0.0.0/8").size == 2**24
        assert ipv4.cidr("192.168.1.0/24").size == 256
        assert ipv4.cidr("0.0.0.0/0").size == 2**32
        assert ipv6.cidr("2001:db8::/32").size == 2**96

    def test_cidr_members(self):
        block = ipv4.cidr("192.168.1.0/30")
        assert [str(a) for a in ipv4.values(block)] == [
            "192.168.1.0", "192.168.1.1", "192.168.1.2", "192.168.1.3",
        ]

    def test_cidr_rejects_the_wrong_family(self):
        with pytest.raises(ValueError):
            ipv4.cidr("2001:db8::/32")

    def test_networks_inverts_cidr(self):
        for text in ("10.0.0.0/8", "0.0.0.0/0", "192.168.1.1/32", "172.16.0.0/12"):
            block = ipv4.cidr(text)
            assert [str(n) for n in ipv4.networks(block)] == [text]

    def test_networks_of_the_empty_set(self):
        assert list(ipv4.networks(ipv4.none)) == []

    @pytest.mark.parametrize(
        "outer, inner",
        [
            ("10.0.0.0/8", "10.1.0.0/16"),
            ("0.0.0.0/0", "127.0.0.0/8"),
            ("192.168.0.0/16", "192.168.1.128/25"),
            ("172.16.0.0/12", "172.16.0.0/12"),
        ],
    )
    def test_minimal_cover_matches_address_exclude(self, outer, inner):
        """`ipaddress` computes the same minimal cover, independently."""
        a, b = ipaddress.ip_network(outer), ipaddress.ip_network(inner)
        got = list(ipv4.networks(ipv4.cidr(outer) - ipv4.cidr(inner)))
        want = sorted(a.address_exclude(b)) if b.subnet_of(a) else [a]
        assert got == want

    def test_the_worked_example(self):
        cover = ipv4.networks(ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16"))
        assert [str(n) for n in cover][:2] == ["10.0.0.0/16", "10.2.0.0/15"]

    blocks = st.sampled_from([
        "10.0.0.0/8", "10.0.0.0/12", "10.1.0.0/16", "10.1.2.0/24", "10.128.0.0/9",
        "192.168.0.0/16", "192.168.1.0/24", "0.0.0.0/1", "172.16.0.0/12",
    ])

    @given(chosen=st.lists(blocks, min_size=1, max_size=4), drop=st.lists(blocks, max_size=3))
    @settings(max_examples=150, deadline=None)
    def test_cover_round_trips_through_any_algebra(self, chosen, drop):
        source = ipv4.none
        for text in chosen:
            source = source | ipv4.cidr(text)
        for text in drop:
            source = source - ipv4.cidr(text)
        cover = list(ipv4.networks(source))
        rebuilt = ipv4.none
        for network in cover:
            rebuilt = rebuilt | ipv4.cidr(str(network))
        assert rebuilt == source
        assert cover == sorted(set(cover))  # minimal: no duplicates, ascending
        assert sum(n.num_addresses for n in cover) == source.size

    @given(chosen=st.lists(blocks, min_size=2, max_size=3))
    @settings(max_examples=100, deadline=None)
    def test_subset_matches_ipaddress(self, chosen):
        a, b = chosen[0], chosen[1]
        assume(a != b)
        networks = ipaddress.ip_network(a), ipaddress.ip_network(b)
        assert (ipv4.cidr(a) <= ipv4.cidr(b)) == networks[0].subnet_of(networks[1])

    def test_range(self):
        span = ipv4.range("10.0.0.5", "10.0.0.9")
        assert [str(a) for a in ipv4.values(span)] == [
            "10.0.0.5", "10.0.0.6", "10.0.0.7", "10.0.0.8",
        ]

    def test_sample_is_always_a_member(self):
        block = ipv4.cidr("10.0.0.0/8")
        rng = random.Random(0)
        for _ in range(100):
            assert ipv4.encode(ipv4.sample(block, rng)) in block

    def test_set_algebra_across_blocks(self):
        private = ipv4.cidr("10.0.0.0/8") | ipv4.cidr("192.168.0.0/16")
        assert private.size == 2**24 + 2**16
        assert ipv4.cidr("10.0.0.0/8") <= private
        assert ipv4.cidr("10.0.0.0/8").isdisjoint(ipv4.cidr("192.168.0.0/16"))
        assert (private & ipv4.cidr("10.1.0.0/16")).size == 2**16


class TestCodecBasics:
    @pytest.mark.parametrize("codec", [float16, float32, float64, ipv4, ipv6])
    def test_all_and_none(self, codec):
        assert codec.all.size == 1 << codec.width
        assert codec.none.size == 0
        assert codec.all.width == codec.width == codec.none.width
        assert ~codec.all == codec.none

    def test_sample_defaults_to_the_whole_universe(self):
        assert isinstance(float16.sample(rng=random.Random(0)), float)
        assert ipv4.sample(rng=random.Random(0)) in ipaddress.ip_network("0.0.0.0/0")

    def test_sample_of_an_empty_set_raises(self):
        with pytest.raises(KeyError):
            float16.sample(float16.none)

    def test_results_are_plain_intsets(self):
        assert isinstance(float64.nan, IntSet)
        assert isinstance(ipv4.cidr("10.0.0.0/8"), IntSet)
