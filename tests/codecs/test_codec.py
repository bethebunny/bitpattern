"""Codec and BDDSet, which every codec shares."""

import ipaddress
import math
import operator
import random
from enum import Enum, Flag
from functools import cached_property

import pytest

from bitpattern import IntSet, Pattern
from bitpattern.codecs import (
    BDDSet,
    Flags,
    Int,
    Members,
    OrderedCodec,
    Tuple,
    bfloat16,
    char,
    dates,
    datetimes,
    float8_e4m3fn,
    float16,
    float32,
    float64,
    int8,
    ipv4,
    ipv6,
    mac,
    naive_datetimes,
    uint128,
    uuid128,
)

SMALL = float16.range(1.0, 1.01)  # the 11 float16s in [1.0, 1.01)


class Size(Enum):
    S = "s"
    M = "m"
    L = "l"


class Style(Flag):
    BOLD = 1
    ITALIC = 2


# One of each kind of codec.
CODECS = [
    char,
    int8,
    uint128,
    float16,
    float8_e4m3fn,
    ipv4,
    ipv6,
    mac,
    uuid128,
    dates,
    datetimes,
    naive_datetimes,
    Members(Size),
    Flags(Style),
    Tuple(int8, char),
]


@pytest.mark.parametrize("codec", CODECS, ids=repr)
def test_codecs_refuse_other_types_with_type_error(codec):
    with pytest.raises(TypeError):
        codec.encode(object())
    assert object() not in codec.all


# Decoding bits outside `all` doesn't have to fail, but these codecs fail past
# their width.
FAILING_DECODES = [
    char,
    float16,
    ipv4,
    ipv6,
    mac,
    uuid128,
    dates,
    datetimes,
    naive_datetimes,
    Members(Size),
    Flags(Style),
]


@pytest.mark.parametrize("codec", FAILING_DECODES, ids=repr)
def test_decode_failures_are_index_errors(codec):
    with pytest.raises(IndexError):
        codec.decode(1 << codec.width)


def test_pattern_has_to_be_the_codecs_width():
    assert float16.pattern("?111.11??.????.????").size == float16.nan.size + 2
    with pytest.raises(ValueError):
        float16.pattern("0000")


def test_set_round_trips():
    assert sorted(float16.set([1.0, -2.0, 0.5])) == [-2.0, 0.5, 1.0]


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
    assert len(values) == len(SMALL) == SMALL.size == 11
    assert values == sorted(values) and all(1.0 <= v < 1.01 for v in values)
    assert [SMALL[at] for at in range(11)] == values
    assert SMALL[-1] == values[-1]
    assert list(reversed(SMALL)) == values[::-1]
    assert [SMALL.index(v) for v in values] == list(range(11))
    assert SMALL.count(values[3]) == 1 and SMALL.count(2.0) == 0
    with pytest.raises(ValueError):
        SMALL.index(2.0)
    with pytest.raises(ValueError):  # like list.index, even for another type
        SMALL.index("x")


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


class Digit(OrderedCodec[int]):
    """0 to 9, which leaves 6 of its 16 bit patterns unused."""

    def encode(self, value: object) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{value!r} isn't an int")
        if not 0 <= value <= 9:
            raise ValueError(f"{value!r} isn't a digit")
        return value

    def decode(self, bits: int) -> int:
        return bits

    @cached_property
    def all(self) -> BDDSet[int]:
        return self.set(range(10))


def test_ordered_codecs_have_a_min_a_max_and_ranges():
    digit = Digit(4, "digit")
    assert (digit.min, digit.max) == (0, 9)
    assert list(digit.range(3, 6)) == [3, 4, 5]
    assert list(digit.range(7)) == [7, 8, 9]
    assert list(digit.range(stop=2)) == [0, 1]
    assert digit.range() == digit.all
    assert digit.range(6, 3) == digit.none


def test_patterns_stay_inside_the_codecs_values():
    assert list(Digit(4, "digit").pattern("1*")) == [8, 9]


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


def test_bdd_set_reprs_write_ints_too_long_for_decimal_in_hex():
    wide, top = Int(20_000, "u20000", signed=False), 1 << 19_999
    assert repr(wide.set([1, top])) == f"BDDSet(u20000, [1, {top:#x}])"
    many = wide.set([*range(10), top])
    assert repr(many) == f"BDDSet(u20000, [0, 1, 2, 3, ..., 9, {top:#x}], size=11)"
    pairs, alone = Tuple(int8, wide), Tuple(wide)
    written = f"BDDSet({pairs}, [(1, {top:#x}), (2, 3)])"
    assert repr(pairs.set([(1, top), (2, 3)])) == written
    assert repr(alone.set([(top,)])) == f"BDDSet({alone}, [({top:#x},)])"


def test_big_bdd_sets_dont_go_through_len():
    assert float64.finite
    assert float64.finite.size == 2**64 - 2**53
    assert float64.finite[-1] == -1.7976931348623157e308
    assert next(reversed(float64.finite)) == -1.7976931348623157e308


def test_codec_sets_are_bdd_sets():
    assert type(float64.nan) is BDDSet and type(float64.nan.storage) is IntSet
    assert type(float64.pattern("*" + ".*" * 15).storage) is Pattern


@pytest.mark.parametrize(
    "codec", [float16, bfloat16, float32, float64, int8, uint128, ipv4, ipv6]
)
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
