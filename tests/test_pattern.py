"""`Pattern` parsing, and the sets patterns describe, against brute force."""

import functools
import itertools
import operator
import pickle
from collections.abc import Set as AbstractSet

import pytest

from bitpattern import IntSet, Pattern
from bitpattern.bdd import node_count
from bitpattern.codecs import ipv4
from bitpattern.pattern import dotted

# The pattern the interface was designed around.
TARGET = "*1.*.*.0000.1111.?01?"

NARROW = Pattern("?1")  # {1, 3}
WIDE = Pattern("00??")  # {0, 1, 2, 3}


def matches(bits: str, value: int) -> bool:
    """Whether `value` matches an expanded pattern, checked bit by bit."""
    width = len(bits)
    return all(
        char == "?" or int(char) == (value >> (width - 1 - at)) & 1
        for at, char in enumerate(bits)
    )


def test_target_expansion():
    assert Pattern(TARGET).bits == "???1????????00001111?01?"


def test_target_shape():
    pattern = Pattern(TARGET)
    assert pattern.width == 24
    assert pattern.free == 13


def test_target_membership():
    pattern = Pattern(TARGET)
    assert len(pattern) == 1 << 13 == 8192
    assert pattern[0] == 1048818
    assert pattern[-1] == 16773371
    assert pattern[0] in pattern
    assert pattern[-1] in pattern
    assert 0 not in pattern


def test_target_costs_one_node_per_pinned_bit():
    assert node_count(Pattern(TARGET).bdd) == 24 - 13


@pytest.mark.parametrize(
    "text, bits",
    [
        ("*", "????"),
        ("*1", "???1"),
        ("1*", "1???"),
        ("1*1", "1??1"),
        ("??*?", "????"),
        ("0000", "0000"),
        ("?01?", "?01?"),
        ("*.*", "????????"),
        ("*1.0000", "???10000"),
        (TARGET, "???1????????00001111?01?"),
    ],
)
def test_glob_fills_its_own_group(text, bits):
    assert Pattern(text).bits == bits


@pytest.mark.parametrize(
    "text, width",
    [("1", 1), ("01", 2), ("?01", 3), ("0000", 4), ("1.0000", 5), ("?01.1111", 7)],
)
def test_leading_group_can_be_short(text, width):
    assert Pattern(text).width == width


@pytest.mark.parametrize(
    "text, members",
    [
        ("1", [1]),
        ("01", [1]),
        ("?1", [1, 3]),
        ("1.0000", [16]),
        ("?01.1111", [31, 95]),
        ("0.????", list(range(16))),
        ("1.????", list(range(16, 32))),
        ("?.0000", [0, 16]),
    ],
)
def test_short_leading_groups(text, members):
    assert list(Pattern(text)) == members


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty pattern
        ".0000",  # empty leading group
        "0000.",  # empty trailing group
        "0000..0000",  # empty middle group
        "*1*",  # ambiguous width
        "00000",  # more than 4 bits
        "*00000",  # more than 4 bits, with a glob
        "0x0",  # not a bit
        "0000.01",  # only the leading group can be short
        "0000.1",
        "*.01",
        "00 00",  # whitespace inside a group
    ],
)
def test_invalid_patterns(text):
    with pytest.raises(ValueError):
        Pattern(text)


def test_error_names_the_group():
    with pytest.raises(ValueError, match=r"'0x0'"):
        Pattern("0x0.0000")


@pytest.mark.parametrize("width", range(1, 7))
def test_agrees_with_matching_bit_by_bit(width):
    """Every pattern of `01?` up to 6 bits wide. Widths 1, 2, 3, 5 and 6 aren't
    multiples of 4, so this covers short leading groups too."""
    universe = range(1 << width)
    for combination in itertools.product("01?", repeat=width):
        bits = "".join(combination)
        pattern = Pattern(dotted(bits))
        expected = [value for value in universe if matches(bits, value)]

        assert pattern.bits == bits
        assert pattern.width == width
        assert len(pattern) == len(expected) == 1 << bits.count("?")
        assert list(pattern) == expected
        assert [pattern[at] for at in range(len(expected))] == expected
        assert [value in pattern for value in universe] == [
            value in expected for value in universe
        ]


@pytest.mark.parametrize(
    "text, canonical",
    [
        (TARGET, "???1.????.????.0000.1111.?01?"),
        ("*", "????"),
        ("1", "1"),
        ("1.0000", "1.0000"),
        ("?01.1111", "?01.1111"),
    ],
)
def test_str_is_the_expanded_spelling(text, canonical):
    assert str(Pattern(text)) == canonical
    assert repr(Pattern(text)) == f"Pattern({canonical!r})"


@pytest.mark.parametrize("width", range(1, 7))
def test_str_round_trips(width):
    for combination in itertools.product("01?", repeat=width):
        pattern = Pattern(dotted("".join(combination)))
        assert Pattern(str(pattern)) == pattern


def test_spelling_doesnt_affect_equality():
    assert Pattern("*") == Pattern("????")
    assert Pattern("*1") == Pattern("???1")
    assert Pattern("*") != Pattern("?.????")
    assert Pattern("*") != "????"


def test_patterns_are_hashable():
    assert len({Pattern("*"), Pattern("????"), Pattern("*1")}) == 2


def test_operators_across_widths():
    assert set(NARROW & WIDE) == {1, 3}
    assert set(NARROW | WIDE) == {0, 1, 2, 3}
    assert set(NARROW - WIDE) == set()
    assert set(WIDE - NARROW) == {0, 2}


def test_result_takes_the_wider_width():
    assert (NARROW & WIDE).width == 4
    assert (NARROW | WIDE).width == 4


def test_operators_take_plain_iterables():
    assert set(WIDE & [1, 2, 99]) == {1, 2}
    assert set(WIDE & IntSet([3])) == {3}


def test_disjoint_patterns():
    assert set(Pattern("0000") & Pattern("1111")) == set()
    assert set(Pattern("0000") | Pattern("1111")) == {0, 15}


def test_patterns_are_closed_under_the_algebra():
    a, b = Pattern("00??"), Pattern("0?0?")
    for result in (a & b, a | b, a - b, ~a):
        assert isinstance(result, Pattern)
        assert result.width == 4
    assert set(a & b) == {0, 1}
    assert set(a | b) == {0, 1, 2, 3, 4, 5}
    assert set(a - b) == {2, 3}
    assert set(~a) == set(range(4, 16))


def test_patterns_are_sets():
    assert isinstance(Pattern("00??"), AbstractSet)
    assert Pattern("00??") <= Pattern("0???")
    assert Pattern("0000").isdisjoint(Pattern("1111"))
    assert set(Pattern("00??")) == {0, 1, 2, 3}


@pytest.mark.parametrize(
    "text",
    ["0000 | 1111", "0000|1111", "0000 | 0001 | 0010", "*|*", "0000 |", "| 0000"],
)
def test_patterns_cant_contain_unions(text):
    """Unions are Python, not pattern syntax."""
    with pytest.raises(ValueError, match=r"can't contain '\|'"):
        Pattern(text)


def test_union_error_shows_how_to_write_it():
    with pytest.raises(ValueError, match=r"Pattern\('0000'\) \| Pattern\('1111'\)"):
        Pattern("0000 | 1111")


@pytest.mark.parametrize(
    "built, members",
    [
        (lambda: Pattern("0000") | Pattern("1111"), [0, 15]),
        (lambda: Pattern("000?") | Pattern("1111"), [0, 1, 15]),
        (lambda: Pattern("0000") | Pattern("0001") | Pattern("0010"), [0, 1, 2]),
        (lambda: ~Pattern("00??"), list(range(4, 16))),
        (lambda: Pattern("0???") - Pattern("00??"), [4, 5, 6, 7]),
    ],
)
def test_operators_build_unions(built, members):
    pattern = built()
    assert isinstance(pattern, Pattern)
    assert list(pattern) == members


@pytest.mark.parametrize(
    "spelling, canonical",
    [
        (lambda: Pattern("0000") | Pattern("0001"), "000?"),
        (lambda: Pattern("0001") | Pattern("0000") | Pattern("0000"), "000?"),
        (lambda: Pattern("000?") | Pattern("0001"), "000?"),
        (lambda: Pattern("0???") | Pattern("1???"), "????"),
    ],
)
def test_unions_are_canonical(spelling, canonical):
    """Branches come from the diagram, so the spelling doesn't survive."""
    built = spelling()
    assert built == Pattern(canonical)
    assert str(built) == canonical
    assert built.branches == (canonical,)
    assert hash(built) == hash(Pattern(canonical))


def test_union_repr_is_the_expression_that_builds_it():
    pattern = ~Pattern("00??")
    assert repr(pattern) == "Pattern('01??') | Pattern('1???')"
    assert eval(repr(pattern)) == pattern


def test_union_str_joins_the_branches():
    assert str(~Pattern("00??")) == "01?? | 1???"


def test_long_unions_are_summarized():
    spare = (ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")).storage
    pattern = Pattern.from_bdd(spare.bdd, spare.width)
    assert repr(pattern) == "<Pattern: 8 branches, width=32, size=16711680>"


@pytest.mark.parametrize("width", range(1, 5))
def test_branches_rebuild_every_subset(width):
    """Every subset of a small universe, written out as branches and read back."""
    for mask in range(1 << (1 << width)):
        members = {v for v in range(1 << width) if mask >> v & 1}
        pattern = Pattern.from_bdd(IntSet(members, width=width).bdd, width)
        branches = [Pattern(branch) for branch in pattern.branches]
        assert all(branch.width == width for branch in branches)
        # Branches are disjoint, so their sizes add up to the whole.
        assert sum(branch.size for branch in branches) == len(members)
        assert (
            functools.reduce(operator.or_, branches, IntSet([], width=width)) == pattern
        )


def test_free_bits_dont_branch():
    """Unlike a CIDR block, a pattern can leave any bit free."""
    odd = Pattern("????.????.????.???1")
    assert odd.branches == ("????.????.????.???1",)
    assert odd.size == 1 << 15


def test_empty_and_full():
    empty = ~Pattern("????")
    assert empty.branches == () and empty.size == 0 and list(empty) == []
    assert Pattern("????").branches == ("????",)
    assert ~empty == Pattern("????")


def test_bits_and_free_need_one_branch():
    union = Pattern("0000") | Pattern("1111")
    for name in ("bits", "free"):
        with pytest.raises(ValueError, match="2 branches"):
            getattr(union, name)


def test_patterns_pickle():
    pattern = ~Pattern("00??")
    restored = pickle.loads(pickle.dumps(pattern))
    assert type(restored) is Pattern and restored == pattern


def test_any_intset_can_be_read_as_a_pattern():
    source = IntSet([1, 4, 9, 16, 25], width=8)
    pattern = Pattern.from_bdd(source.bdd, source.width)
    assert type(pattern) is Pattern and pattern == source
