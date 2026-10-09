"""The char codec, checked against unicodedata over every code point."""

import collections
import string
import sys
import unicodedata
from functools import cache

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern import IntSet
from bitpattern.codecs.char import category_points, char

SURROGATES = range(0xD800, 0xE000)


@cache
def categories() -> list[str]:
    """Each code point's general category, by brute force."""
    return [unicodedata.category(chr(point)) for point in range(sys.maxunicode + 1)]


@cache
def counts() -> collections.Counter[str]:
    """How many characters are in each general category."""
    return collections.Counter(
        name for point, name in enumerate(categories()) if point not in SURROGATES
    )


CATEGORIES = sorted(set(categories()))
MAJOR_CLASSES = sorted({name[0] for name in CATEGORIES})

# Ranges change at their bounds and at the surrogates, so probe either side of each.
BOUNDS = [0, 0x61, 0x7B, 0xD7FF, 0xE000, 0x10000, sys.maxunicode]
EDGES = [*BOUNDS, SURROGATES[0], SURROGATES[-1]]
PROBES = [point for edge in EDGES for point in (edge - 1, edge) if point >= 0]

CODE_POINTS = st.integers(0, sys.maxunicode)
SCALAR_VALUES = CODE_POINTS.filter(lambda point: point not in SURROGATES)


def test_chars_are_every_code_point_but_the_surrogates():
    scalars = (point for point in range(sys.maxunicode + 1) if point not in SURROGATES)
    for point, value in zip(scalars, char.all, strict=True):
        assert value == chr(point)
        assert char.encode(value) == point
    assert char.all.size == sys.maxunicode + 1 - len(SURROGATES)


def test_surrogates_arent_chars():
    for point in SURROGATES:
        assert chr(point) not in char.all
        with pytest.raises(ValueError):
            char.encode(chr(point))


@pytest.mark.parametrize("value", ["", "ab", "e\u0301", 97, None, b"a", ["a"]])
def test_chars_refuse_what_isnt_one_character(value):
    with pytest.raises(ValueError):
        char.encode(value)
    assert value not in char.all


def test_patterns_only_match_chars():
    assert char.pattern("?.*.*.*.*.*") == char.all
    assert char.pattern("0.0000.1101.1*.*.*") == char.none  # the surrogates
    assert char.pattern("1.0001.*.*.*.*") == char.none  # past U+10FFFF
    assert char.pattern("0.0000.0000.0000.01?0.0001") == char.set("Aa")  # bit 5 is case


def test_invert_stays_inside_all():
    letters = char.range("a", "{")
    assert (~letters).size == char.all.size - 26
    assert (~letters).storage.isdisjoint(IntSet.range(0xD800, 0xE000))
    assert (~letters).storage[-1] == sys.maxunicode
    assert ~~letters == letters


def test_char_range_is_half_open():
    assert list(char.range("a", "{")) == list(string.ascii_lowercase)
    assert char.range("a", "a") == char.none
    assert char.range("b", "a") == char.none


def test_char_range_skips_the_surrogates():
    around = char.range("\ud7fe", "\ue002")
    assert list(around) == ["\ud7fe", "\ud7ff", "\ue000", "\ue001"]


def test_surrogates_cant_bound_a_char_range():
    with pytest.raises(ValueError):
        char.range("\ud800", "\ue000")
    with pytest.raises(ValueError):
        char.range(stop="\udfff")


@pytest.mark.parametrize("low", BOUNDS)
def test_char_range_agrees_with_brute_force(low):
    for high in BOUNDS:
        chars, points = char.range(chr(low), chr(high)), range(low, high)
        assert chars.size == len(points) - sum(point in points for point in SURROGATES)
        inside = {point for point in PROBES if chr(point) in chars}
        assert inside == {
            point for point in PROBES if point in points and point not in SURROGATES
        }


@given(low=SCALAR_VALUES, high=SCALAR_VALUES, point=CODE_POINTS)
@settings(deadline=None)
def test_char_range_holds_the_chars_between_its_bounds(low, high, point):
    inside = low <= point < high and point not in SURROGATES
    assert (chr(point) in char.range(chr(low), chr(high))) == inside


@pytest.mark.parametrize("name", [*CATEGORIES, *MAJOR_CLASSES])
def test_categories_agree_with_unicodedata(name):
    chars = char.category(name)
    expected = sum(n for category, n in counts().items() if category.startswith(name))
    assert chars.size == expected
    assert all(
        point not in SURROGATES and categories()[point].startswith(name)
        for point in chars.storage
    )


def test_surrogates_are_in_cs_but_arent_chars():
    assert unicodedata.category("\ud800") == "Cs"
    assert char.category("Cs") == char.none


@pytest.mark.parametrize("name", ["", "Xx", "lu", "Letter"])
def test_category_refuses_unknown_names(name):
    with pytest.raises(ValueError):
        char.category(name)


def test_categories_are_cached():
    assert category_points("Lu", char.width) is category_points("Lu", char.width)


def test_char_ranges_can_leave_an_end_open():
    assert char.range() == char.all
    assert char.range(stop="\x80") == char.range("\0", "\x80")
    assert list(char.range("\U0010fffe")) == ["\U0010fffe", "\U0010ffff"]
    assert (char.min, char.max) == ("\0", "\U0010ffff")
