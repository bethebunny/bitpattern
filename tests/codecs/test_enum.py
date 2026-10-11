"""Enum and flag codecs, brute-forced over small enums."""

import itertools
from enum import KEEP, Enum, Flag, IntEnum, IntFlag

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern.codecs.enum import Flags, Members


class Suit(Enum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"


class Color(Enum):
    """Three members in two bits, defined against the order of their values."""

    RED = 3
    GREEN = 2
    BLUE = 1
    CRIMSON = RED  # an alias


class Planet(IntEnum):
    MERCURY = 1
    VENUS = 2
    EARTH = 3
    MARS = 4
    JUPITER = 5


class Single(Enum):
    ONLY = "only"


class Perm(Flag):
    R = 4
    W = 2
    X = 1
    NONE = 0
    RWX = 7  # made of the other flags, so RWX adds no combinations


class Mode(IntFlag):
    """Nothing uses bit 2. IntFlag keeps bits that aren't flags, so Mode(4) exists."""

    READ = 1
    WRITE = 2
    APPEND = 8


class Wide(Flag, boundary=KEEP):
    """BC is two bits that no other flag has, so both bits are set or neither.
    KEEP lets Wide(2) exist anyway."""

    A = 1
    BC = 6


class Overlapping(Flag, boundary=KEEP):
    """Flags that share a bit, without any single-bit flags. KEEP lets Overlapping(5)
    exist, though 5 isn't a combination."""

    AB = 3
    BC = 6


MEMBERS = [Suit, Color, Planet, Single]
FLAGS = [Perm, Mode, Wide, Overlapping]

Many = IntFlag("Many", [f"F{at}" for at in range(100)])
Sparse = IntFlag("Sparse", {f"F{at}": 1 << 2 * at for at in range(50)})
ODD = sum(1 << 2 * at + 1 for at in range(50))  # the bits between Sparse's flags


def subsets(values):
    """Every subset of `values`, as tuples."""
    return itertools.chain.from_iterable(
        itertools.combinations(values, size) for size in range(len(values) + 1)
    )


def combinations(flag) -> set[int]:
    """The value of every combination of `flag`'s members, adding one at a time."""
    found = {0}
    for member in flag.__members__.values():
        found |= {value | member.value for value in found}
    return found


@pytest.mark.parametrize("enum", MEMBERS)
def test_members_encode_as_their_index_in_definition_order(enum):
    members = Members(enum)
    assert [members.encode(member) for member in enum] == list(range(len(enum)))
    assert [members.decode(at) for at in range(len(enum))] == list(enum)


def test_aliases_arent_separate_members():
    assert Members(Color).encode(Color.CRIMSON) == 0
    assert Members(Color).all.size == 3


def test_member_patterns_leave_out_unused_indices():
    assert list(Members(Color).pattern("1?")) == [Color.BLUE]  # and not index 3


@pytest.mark.parametrize("count", range(1, 18))
def test_members_take_as_few_bits_as_number_them(count):
    width = Members(Enum("Counted", [f"M{at}" for at in range(count)])).width
    assert 1 << width >= count > (1 << width) // 2


@pytest.mark.parametrize("enum", MEMBERS)
def test_members_all_is_every_member_in_definition_order(enum):
    assert list(Members(enum).all) == list(enum)


@pytest.mark.parametrize("enum", MEMBERS)
def test_invert_stays_inside_the_members(enum):
    members = Members(enum)
    for chosen in subsets(list(enum)):
        rest = [member for member in enum if member not in chosen]
        assert ~members.set(chosen) == members.set(rest)


@pytest.mark.parametrize("flag", FLAGS)
def test_flags_encode_as_their_own_bits(flag):
    flags = Flags(flag)
    for member in flag.__members__.values():
        assert flags.encode(member) == member.value
    for value in combinations(flag):
        assert flags.encode(flag(value)) == value
        assert flags.decode(value) is flag(value)


@pytest.mark.parametrize("flag", FLAGS)
def test_flags_all_is_every_combination_in_order_of_value(flag):
    assert [value.value for value in Flags(flag).all] == sorted(combinations(flag))


@pytest.mark.parametrize("flag", FLAGS)
def test_invert_stays_inside_the_combinations(flag):
    flags, values = Flags(flag), sorted(combinations(flag))
    for chosen in subsets(values):
        rest = [value for value in values if value not in chosen]
        assert ~flags.set(map(flag, chosen)) == flags.set(map(flag, rest))


def test_flags_are_as_wide_as_their_widest_flag():
    assert [Flags(flag).width for flag in FLAGS] == [3, 4, 3, 3]


def test_flag_patterns_match_their_bits():
    perms = Flags(Perm)
    assert list(perms.pattern("1?0")) == [Perm.R, Perm.R | Perm.W]
    assert perms.pattern("1??") == perms.having(Perm.R)
    assert perms.pattern("?0?") == perms.lacking(Perm.W)


@pytest.mark.parametrize("flag", FLAGS)
def test_having_and_lacking_agree_with_brute_force(flag):
    flags, found = Flags(flag), combinations(flag)
    for value in found:
        having = {other for other in found if other & value == value}
        lacking = {other for other in found if not other & value}
        assert set(flags.having(flag(value)).storage) == having
        assert set(flags.lacking(flag(value)).storage) == lacking


@pytest.mark.parametrize(
    "codec, value, error",
    [
        (Members(Suit), Color.RED, TypeError),  # another enum's member
        (Members(Suit), "c", TypeError),  # a member's value
        (Members(Planet), 1, TypeError),  # == Planet.MERCURY, but not a member
        (Members(Perm), Perm.R | Perm.W, ValueError),  # a combination, not a member
        (Flags(Perm), Mode.READ, TypeError),  # another flag's member
        (Flags(Mode), 1, TypeError),  # == Mode.READ, but not a member
        (Flags(Mode), Mode(4), ValueError),  # a bit that isn't any flag's
        (Flags(Mode), Mode(16), ValueError),  # past the widest flag
        (Flags(Wide), Wide(2), ValueError),  # half of BC
        (Flags(Overlapping), Overlapping(5), ValueError),  # needs a bit from each flag
    ],
)
def test_enum_codecs_refuse_what_they_cant_encode(codec, value, error):
    with pytest.raises(error):
        codec.encode(value)
    assert value not in codec.all


def test_enum_decode_failures_are_index_errors():
    with pytest.raises(IndexError):
        Members(Color).decode(3)  # three members take two bits
    with pytest.raises(IndexError):
        Flags(Perm).decode(8)  # a strict Flag refuses bits outside every flag


def test_enum_codecs_over_the_same_enum_are_equal():
    assert Members(Suit) == Members(Suit) and hash(Members(Suit)) == hash(Members(Suit))
    assert Flags(Perm) == Flags(Perm) and hash(Flags(Perm)) == hash(Flags(Perm))
    hearts, spades = Members(Suit).set([Suit.HEARTS]), Members(Suit).set([Suit.SPADES])
    assert list(hearts | spades) == [Suit.HEARTS, Suit.SPADES]


@pytest.mark.parametrize(
    "a, b",
    [
        (Members(Suit), Members(Color)),
        (Members(Perm), Flags(Perm)),
        (Flags(Perm), Flags(Mode)),
    ],
)
def test_enum_codecs_dont_combine_with_other_codecs(a, b):
    assert a != b
    with pytest.raises(TypeError):
        a.all | b.all


def test_enum_codecs_are_named_after_their_enum():
    assert repr(Members(Suit)) == "Suit"
    assert repr(Flags(Perm)) == "Perm"


def test_many_flags_arent_enumerated():
    many = Flags(Many)
    assert many.width == 100
    assert many.all.size == 2**100
    assert many.having(many.decode(1 << 99 | 1)).size == 2**98
    assert Flags(Sparse).all.size == 2**50


@given(st.integers(0, 2**100 - 1))
@settings(deadline=None)
def test_many_flags_round_trip(value):
    many = Flags(Many)
    assert many.encode(many.decode(value)) == value


@given(st.integers(0, 2**99 - 1))
@settings(deadline=None)
def test_sparse_flags_refuse_the_bits_between_them(value):
    assert (Sparse(value) in Flags(Sparse).all) == (not value & ODD)
