"""`IntSet` cross-checked against the builtin `set`."""

import copy
import itertools
import operator
import pickle
import random
from collections.abc import MutableSequence, MutableSet, Sequence
from collections.abc import Set as AbstractSet
from typing import NamedTuple

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bitpattern.bdd import BDD, interval, node_count
from bitpattern.intset import IntSet, written
from bitpattern.pattern import Pattern

# Every subset of a three-bit universe, which is few enough to check exhaustively.
WIDTH = 3
UNIVERSE = range(1 << WIDTH)
MASKS = range(1 << len(UNIVERSE))


def subset(mask):
    return {value for value in UNIVERSE if mask >> value & 1}


SUBSETS = [subset(mask) for mask in MASKS]
SETS = [IntSet(members, width=WIDTH) for members in SUBSETS]
NODES = [group.bdd for group in SETS]

VALUES = st.integers(min_value=0, max_value=(1 << 16) - 1)
GROUPS = st.sets(VALUES, max_size=40)

HUGE = IntSet.range(0, 1 << 64, width=64)


def samples(width: int, count: int = 12) -> list[set[int]]:
    """A spread of subsets of `range(2 ** width)`, or every subset if there are few."""
    size = 1 << width
    if size <= 4:
        return [{v for v in range(size) if mask >> v & 1} for mask in range(1 << size)]
    chooser = random.Random(width)
    edges = [set(), set(range(size)), {0}, {size - 1}, {0, size - 1}]
    drawn = [
        set(chooser.sample(range(size), chooser.randrange(size))) for _ in range(count)
    ]
    return edges + drawn


def check(group: IntSet, members: set[int]):
    """Every way of reading `group` has to agree with `members`."""
    expected = sorted(members)
    assert len(group) == len(expected)
    assert list(group) == expected
    for value in range(1 << group.width):
        assert (value in group) == (value in members)
    for at, value in enumerate(expected):
        assert group[at] == value
        assert group[at - len(expected)] == value
    for at in (len(expected), -len(expected) - 1):
        with pytest.raises(IndexError):
            group[at]


@pytest.mark.parametrize("mask", MASKS)
def test_reads_agree_with_builtin_set(mask):
    check(SETS[mask], SUBSETS[mask])


@pytest.mark.parametrize("left", MASKS)
def test_operators_agree_with_builtin_set(left):
    """All 256x256 pairs. Diagrams are canonical, so equal means identical."""
    for right in MASKS:
        assert (SETS[left] & SETS[right]).bdd is NODES[left & right]
        assert (SETS[left] | SETS[right]).bdd is NODES[left | right]
        assert (SETS[left] - SETS[right]).bdd is NODES[left & ~right & 0xFF]


@pytest.mark.parametrize("left", MASKS)
def test_equality_matches_membership(left):
    for right in MASKS:
        assert (SETS[left] == SETS[right]) == (left == right)


@given(members=GROUPS)
@settings(max_examples=200, deadline=None)
def test_reads_agree_with_builtin_set_at_larger_widths(members):
    group = IntSet(members)
    assert len(group) == len(members)
    assert list(group) == sorted(members)
    for at, value in enumerate(sorted(members)):
        assert group[at] == value
        assert value in group


@pytest.mark.parametrize("width", [16, 200])
def test_iterating_runs_and_lone_members(width):
    """Iterating hands over long runs of members whole, and the rest in lists, which
    start at 16 members and double up to 1024."""
    members = [1, 3, *range(10, 3000), 4000, *range(5000, 6024), *range(7000, 8023)]
    members.append(1 << (width - 1))
    assert list(IntSet(members, width=width)) == members
    for size in (15, 16, 17, 31, 33, 63, 64, 65, 1023, 1024, 1025, 4000):
        spaced = range(0, 2 * size, 2)  # no runs, so every member goes in a list
        assert list(IntSet(spaced, width=width)) == list(spaced)


@given(left=GROUPS, right=GROUPS)
@settings(max_examples=200, deadline=None)
def test_operators_agree_with_builtin_set_at_larger_widths(left, right):
    a, b = IntSet(left), IntSet(right)
    assert set(a & b) == left & right
    assert set(a | b) == left | right
    assert set(a - b) == left - right
    assert (a == b) == (left == right)


@given(members=GROUPS, extra=VALUES)
@settings(max_examples=200, deadline=None)
def test_union_then_difference_restores(members, extra):
    assume(extra not in members)
    group = IntSet(members)
    assert set(group | {extra}) == members | {extra}
    assert set((group | {extra}) - {extra}) == members


@given(members=st.sets(VALUES, min_size=1, max_size=40))
@settings(max_examples=200, deadline=None)
def test_indexing_matches_sorted_order(members):
    group = IntSet(members)
    ordered = sorted(members)
    for at in range(-len(ordered), len(ordered)):
        assert group[at] == ordered[at]
    with pytest.raises(IndexError):
        group[len(ordered)]


@pytest.mark.parametrize("narrow", [0, 1, 2, 3])
@pytest.mark.parametrize("wide", [4, 5, 8])
@pytest.mark.parametrize("members", [set(), {0}, {1}, {0, 1}, {0, 3}, {2, 3}])
def test_widening_keeps_the_members(narrow, wide, members):
    members = {value for value in members if value.bit_length() <= narrow}
    group = IntSet(members, width=narrow).widened(wide)
    assert group.width == wide
    check(group, members)


@given(members=GROUPS, width=st.integers(min_value=0, max_value=24))
@settings(max_examples=200, deadline=None)
def test_widening_keeps_the_members_at_larger_widths(members, width):
    group = IntSet(members)
    group = group.widened(group.width + width)
    assert set(group) == members
    assert len(group) == len(members)


def test_widening_doesnt_add_members():
    """`{0}` at width 1 is "bit 0 is clear", ie. every even number."""
    assert list(IntSet([0], width=1).widened(4)) == [0]


@pytest.mark.parametrize("left_width", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("right_width", [0, 1, 2, 3, 5])
def test_operators_across_widths(left_width, right_width):
    width = max(left_width, right_width)
    for left in samples(left_width):
        a = IntSet(left, width=left_width)
        for right in samples(right_width):
            b = IntSet(right, width=right_width)
            for got, want in [
                (a & b, left & right),
                (a | b, left | right),
                (a - b, left - right),
            ]:
                assert got.width == width
                assert set(got) == want
                assert len(got) == len(want)


def test_has_no_mutators():
    for name in ("add", "discard", "clear", "pop", "remove"):
        assert not hasattr(IntSet(), name), name


def test_widened_is_a_new_set():
    group = IntSet([1], width=2)
    wider = group.widened(8)
    assert wider is not group
    assert (group.width, wider.width) == (2, 8)
    assert set(group) == set(wider) == {1}


def test_widened_is_the_same_set_if_already_wide_enough():
    group = IntSet([1], width=8)
    assert group.widened(4) is group


def test_in_place_operators_rebind():
    group = IntSet([1, 2], width=4)
    other = group
    group |= [9]
    assert sorted(group) == [1, 2, 9]
    assert sorted(other) == [1, 2]


def test_rejects_negatives():
    with pytest.raises(ValueError):
        IntSet([-1])


def test_only_non_negative_ints_are_members():
    assert -1 not in IntSet([1], width=4)
    assert "1" not in IntSet([1], width=4)
    assert None not in IntSet([1], width=4)


@pytest.mark.parametrize("mask", MASKS)
def test_equal_sets_hash_equally_across_widths(mask):
    """Equality is by members, so hashing has to ignore the width."""
    forms = [IntSet(SUBSETS[mask], width=width) for width in range(3, 10)]
    assert len({hash(form) for form in forms}) == 1
    assert all(form == forms[0] for form in forms)


@given(members=GROUPS, extra=st.integers(0, 40))
@settings(max_examples=200, deadline=None)
def test_widening_keeps_the_hash(members, extra):
    group = IntSet(members)
    assert hash(group.widened(group.width + extra)) == hash(group)


def test_distinct_sets_hash_apart():
    assert len({hash(group) for group in SETS}) == len(MASKS)


def test_usable_as_dict_keys():
    table = {IntSet([1], width=1): "a", IntSet([1, 2], width=4): "b"}
    assert table[IntSet([1], width=8)] == "a"
    assert table[IntSet([1, 2], width=2)] == "b"


def test_empty_and_zero_hash_apart():
    assert hash(IntSet()) != hash(IntSet([0]))
    assert IntSet() != IntSet([0])


def test_hashing_a_huge_set_is_cheap():
    """`Set._hash` would iterate all 2**64 members."""
    assert isinstance(hash(HUGE), int)


def test_from_bdd():
    group = IntSet.from_bdd(BDD(0, BDD.ACCEPT, BDD.REJECT), 3)
    assert isinstance(group, IntSet)
    assert list(group) == [0, 2, 4, 6]


def test_from_bdd_rejects_a_diagram_wider_than_its_width():
    with pytest.raises(ValueError):
        IntSet.from_bdd(BDD(5, BDD.ACCEPT, BDD.REJECT), 3)


def test_empty_set():
    group = IntSet()
    assert len(group) == 0 and list(group) == [] and 0 not in group


@pytest.mark.parametrize("width", [-1, -100, -(2**100)])
def test_a_negative_width_is_0(width):
    """As for str.ljust, since it's only the least width the set has."""
    assert IntSet(width=width).width == 0
    assert IntSet([5], width=width).width == 3


def test_zero_width_holds_only_zero():
    assert list(IntSet.from_bdd(BDD.ACCEPT, 0)) == [0]
    assert list(IntSet.from_bdd(BDD.REJECT, 0)) == []


def test_operators_take_any_iterable():
    group = IntSet([1, 2, 3], width=2)
    assert set(group & [2, 3, 9]) == {2, 3}
    assert set(group | iter([9])) == {1, 2, 3, 9}
    assert set(group - (2, 3)) == {1}


def test_repr_lists_members_in_order():
    assert repr(IntSet([3, 1], width=2)) == "IntSet([1, 3], width=2)"


@pytest.mark.parametrize("width", range(7))
def test_interval_agrees_with_range(width):
    """Bounds past either end of the width clamp, and never raise."""
    bounds = range(-2, (1 << width) + 2)
    for start in bounds:
        for stop in bounds:
            got = list(IntSet.from_bdd(interval(start, stop, width=width), width))
            assert got == [v for v in range(1 << width) if start <= v < stop]


@pytest.mark.parametrize("width", [126, 127, 128, 129, 200])
def test_intervals_around_the_big_int_widths(width):
    top = 1 << width
    for start, stop in [(0, top), (5, top), (0, top - 3), (top // 2 - 1, top // 2 + 2)]:
        group = IntSet.from_bdd(interval(start, stop, width=width), width)
        assert group.size == stop - start
        assert group[0] == start and group[-1] == stop - 1
        assert start - 1 not in group and stop not in group


@pytest.mark.parametrize("width", range(6))
def test_range_agrees_with_builtin(width):
    for start in range(1 << width):
        for stop in range(1 << width):
            got = IntSet.range(start, stop, width=width)
            assert list(got) == list(range(start, max(start, stop)))
            assert got.width == width


def test_range_width_defaults_to_fit_the_stop():
    assert IntSet.range(3, 17).width == 5
    assert list(IntSet.range(3, 17)) == list(range(3, 17))
    assert IntSet.range(0, 0).width == 0
    assert list(IntSet.range(5, 2)) == []


def test_range_is_one_node_per_bit():
    span = IntSet.range(10**15, 10**18, width=64)
    assert span.size == 10**18 - 10**15
    assert node_count(span.bdd) == 90  # as the README says


def test_range_covers_the_whole_universe():
    assert HUGE.bdd is BDD.ACCEPT
    assert IntSet.range(0, 10**30, width=64).size == 1 << 64


def test_range_rejects_a_negative_start():
    with pytest.raises(ValueError, match="can't hold negative numbers, got -1$"):
        IntSet.range(-1, 5)
    # Python won't write out an int this long.
    with pytest.raises(ValueError, match="got an int of 20001 bits"):
        IntSet.range(-(1 << 20_000), 5)


def test_size_agrees_with_len_when_it_fits():
    group = IntSet([1, 2, 3], width=8)
    assert group.size == len(group) == 3


def test_size_works_where_len_overflows():
    """len() has to fit in a `Py_ssize_t`, and size doesn't."""
    assert HUGE.size == 2**64
    with pytest.raises(OverflowError):
        len(HUGE)


def test_len_overflows_past_2_to_the_63():
    fits = IntSet.range(0, 2**63 - 1, width=64)
    assert fits.size == len(fits)
    with pytest.raises(OverflowError):
        len(IntSet.range(0, 2**63, width=64))


def test_indexing_works_on_a_huge_set():
    assert [HUGE[at] for at in range(3)] == [0, 1, 2]
    assert HUGE[-1] == 2**64 - 1
    assert HUGE[2**63] == 2**63


@pytest.mark.parametrize("mask", MASKS)
def test_complement_agrees_with_builtin_set(mask):
    assert set(~SETS[mask]) == set(UNIVERSE) - SUBSETS[mask]


def test_complement_is_an_involution():
    group = IntSet([1, 5], width=4)
    assert ~~group == group
    assert (group | ~group).bdd is BDD.ACCEPT
    assert (group & ~group).bdd is BDD.REJECT


def test_complement_respects_the_width():
    assert list(~IntSet([1], width=2)) == [0, 2, 3]
    assert list(~IntSet([1], width=3)) == [0, 2, 3, 4, 5, 6, 7]


def test_choice_is_always_a_member():
    group = IntSet([3, 99, 12345], width=16)
    for seed in range(100):
        assert group.choice(rng=random.Random(seed)) in group


def test_choice_reaches_every_member():
    group = IntSet([3, 99, 12345], width=16)
    assert {group.choice(rng=random.Random(seed)) for seed in range(60)} == set(group)


def test_choice_works_where_len_overflows():
    assert HUGE.choice(rng=random.Random(0)) < 2**64
    with pytest.raises(OverflowError):
        random.Random(0).choice(HUGE)


def test_choice_from_an_empty_set_raises():
    with pytest.raises(IndexError):
        IntSet().choice()


def test_random_works_while_len_fits():
    group = IntSet.range(0, 2**62, width=64)
    drawn = random.Random(0).sample(group, 5)
    assert len(set(drawn)) == 5 and all(value in group for value in drawn)
    assert random.Random(0).choice(group) in group
    with pytest.raises(OverflowError):
        random.Random(0).sample(HUGE, 5)


SLICES = [
    slice(None),
    slice(1, None),
    slice(None, -1),
    slice(-3, None),
    slice(2, 5),
    slice(5, 2),
    slice(None, None, 2),
    slice(1, None, 3),
    slice(None, None, -1),
    slice(-2, 0, -2),
]


@pytest.mark.parametrize("mask", MASKS)
def test_slicing_agrees_with_list_slicing(mask):
    """A slice holds the members at the sliced positions, in sorted order."""
    members = sorted(SUBSETS[mask])
    for item in SLICES:
        got = SETS[mask][item]
        assert list(got) == sorted(members[item])
        assert got.width == WIDTH


@given(
    members=GROUPS,
    start=st.none() | st.integers(-50, 50),
    stop=st.none() | st.integers(-50, 50),
    step=st.none() | st.integers(-5, 5).filter(bool),
)
@settings(max_examples=200, deadline=None)
def test_slicing_agrees_with_list_slicing_at_larger_widths(members, start, stop, step):
    group = IntSet(members)
    assert list(group[start:stop:step]) == sorted(sorted(members)[start:stop:step])


def test_slices_keep_the_type_and_width():
    assert IntSet([1, 2, 3], width=8)[1:].width == 8
    assert type(Pattern("00??")[1:3]) is Pattern
    assert Pattern("00??")[1:3] == IntSet([1, 2])


def test_slicing_a_huge_set_doesnt_enumerate():
    assert HUGE[2**62 : 2**63] == IntSet.range(2**62, 2**63, width=64)
    assert list(HUGE[-3:]) == [2**64 - 3, 2**64 - 2, 2**64 - 1]


def position(sequence, value, *bounds):
    """`sequence.index(value, *bounds)`, or None if `value` isn't there."""
    try:
        return sequence.index(value, *bounds)
    except ValueError:
        return None


@pytest.mark.parametrize("mask", MASKS)
def test_index_and_count_agree_with_list(mask):
    group, members = SETS[mask], sorted(SUBSETS[mask])
    for value in [*UNIVERSE, -1, 1 << WIDTH, "1", None]:
        assert group.count(value) == members.count(value)
        for bounds in [(), (1,), (-2,), (0, 2), (1, -1)]:
            assert position(group, value, *bounds) == position(members, value, *bounds)


def test_index_works_on_a_huge_set():
    assert HUGE.index(2**63) == 2**63
    assert HUGE.index(2**64 - 1, -1) == 2**64 - 1
    with pytest.raises(ValueError):
        HUGE.index(2**64)


def test_sets_wider_than_127_bits_are_exact():
    top = (1 << 200) - 1
    group = IntSet([0, 1 << 199, top], width=200)
    assert group.size == 3
    assert list(group) == [0, 1 << 199, top]
    assert group[1] == 1 << 199
    assert group.index(top) == 2
    assert (~group).size == (1 << 200) - 3
    assert top - 1 in ~group


def test_indexing_sets_wider_than_128_bits_keeps_their_high_bits():
    # Multiples of 4 below 2**130: two nodes, under 128 free bits.
    fours = IntSet.from_bdd(BDD(1, BDD(0, BDD.ACCEPT, BDD.REJECT), BDD.REJECT), 130)
    n = 2**126 + 12345
    assert fours[n] == 4 * n
    assert fours.index(4 * n) == n
    assert fours[-1] == 2**130 - 4
    assert fours[n - 2 : n + 2] == IntSet([4 * i for i in range(n - 2, n + 2)])
    assert fours[:] == fours


def test_walking_wide_sets_takes_a_member_of_memory():
    """Iterating keeps one member and a frame per bit, however many bits are free."""
    evens = IntSet.from_bdd(BDD(0, BDD.ACCEPT, BDD.REJECT), 100_000)
    assert list(itertools.islice(evens, 3)) == [0, 2, 4]
    node = BDD(1 << 20, BDD(3, BDD.ACCEPT, BDD.REJECT), BDD.ACCEPT)
    assert next(iter(IntSet.from_bdd(node, (1 << 20) + 1))) == 0


def test_reprs_write_members_too_long_for_decimal_in_hex():
    top = 1 << 19_999
    assert (
        repr(IntSet([0, 5, top], width=20_000))
        == f"IntSet([0, 5, {top:#x}], width=20000)"
    )


class Pair(NamedTuple):
    low: int
    high: int


def test_reprs_leave_a_tuple_subclass_to_its_own_repr():
    """Written out as a plain tuple, a namedtuple would lose its name and fields."""
    assert written(Pair(1, 2)) == "Pair(low=1, high=2)"
    with pytest.raises(ValueError):
        written(Pair(1, 1 << 19_999))


def test_128_bit_sets_reach_both_ends():
    top = 1 << 128
    ends = IntSet([0, 1, 2**127, top - 2, top - 1], width=128)
    assert ends.size == 5
    assert list(ends) == [0, 1, 2**127, top - 2, top - 1]
    assert [ends[i] for i in range(-5, 5)] == list(ends) * 2
    assert [ends.index(value) for value in ends] == list(range(5))
    span = IntSet.range(top - 5, top, width=128)
    assert list(span) == list(range(top - 5, top))
    assert list(reversed(span)) == list(range(top - 1, top - 6, -1))
    everything = IntSet.range(0, top, width=128)
    assert everything.size == top and everything[-1] == top - 1
    assert everything[2**127 + 3] == 2**127 + 3 and everything.index(top - 9) == top - 9
    assert list(itertools.islice(everything, 3)) == [0, 1, 2]


def test_index_of_wide_sets_with_free_high_bits():
    wide = IntSet.from_bdd(IntSet([3, 70, 900], width=16).bdd, 200)
    for high in (0, 1, 2**100, 2**184 - 1):
        for at, low in enumerate([3, 70, 900]):
            assert wide.index(high << 16 | low) == high * 3 + at


def test_the_last_member_of_a_sparse_wide_set():
    sparse = IntSet.from_bdd(IntSet([3, 700], width=11).bdd, 130)
    assert sparse.size == 2**120
    last = (1 << 130) - 2048 | 700
    assert sparse[-1] == next(reversed(sparse)) == last
    assert repr(sparse).endswith(f"{last}], size=2**120, width=130)")


def test_reversed():
    for group, members in zip(SETS, SUBSETS):
        assert list(reversed(group)) == sorted(members, reverse=True)
    assert next(reversed(HUGE)) == 2**64 - 1  # without going through len()


@pytest.mark.parametrize("left", MASKS)
def test_comparisons_agree_with_builtin_set(left):
    a, x = SETS[left], SUBSETS[left]
    for right in MASKS:
        b, y = SETS[right], SUBSETS[right]
        assert (a <= b) == (x <= y)
        assert (a < b) == (x < y)
        assert (a >= b) == (x >= y)
        assert (a > b) == (x > y)
        assert a.isdisjoint(b) == x.isdisjoint(y)
        assert set(a ^ b) == x ^ y


@given(left=GROUPS, right=GROUPS)
@settings(max_examples=200, deadline=None)
def test_comparisons_agree_with_builtin_set_at_larger_widths(left, right):
    a, b = IntSet(left), IntSet(right)
    assert (a <= b) == (left <= right)
    assert (a < b) == (left < right)
    assert a.isdisjoint(b) == left.isdisjoint(right)
    assert set(a ^ b) == left ^ right


def test_comparisons_across_widths():
    assert IntSet([1], width=1) <= IntSet([1, 3], width=2)
    assert not IntSet([1, 3], width=2) <= IntSet([1], width=1)
    assert IntSet([1], width=1).isdisjoint(IntSet([2, 3], width=2))
    assert not IntSet([2], width=2) <= IntSet([0], width=1)  # {0} isn't every even
    assert IntSet([0], width=1) <= IntSet([0, 2], width=2)


def test_is_an_immutable_set_and_sequence():
    assert isinstance(IntSet(), AbstractSet)
    assert isinstance(IntSet(), Sequence)
    assert not isinstance(IntSet(), MutableSet)
    assert not isinstance(IntSet(), MutableSequence)


def test_only_compares_with_intsets():
    """Being equal to a frozenset would mean hashing like one."""
    assert IntSet([1, 2], width=4) != frozenset({1, 2})
    assert frozenset({1, 2}) != IntSet([1, 2], width=4)
    for compare in (operator.le, operator.lt, operator.ge, operator.gt):
        with pytest.raises(TypeError):
            compare(IntSet([1, 2], width=4), {1, 2})


def test_equal_objects_hash_equally():
    values = [
        IntSet([1, 2], width=4),
        IntSet([1, 2], width=8),
        Pattern("00?1"),
        frozenset({1, 2}),
        frozenset({1, 3}),
        IntSet([1, 3], width=2),
    ]
    for a in values:
        for b in values:
            if a == b:
                assert hash(a) == hash(b), (a, b)


def test_antisymmetry():
    a, b = IntSet([1, 2], width=2), IntSet([1, 2], width=8)
    assert a <= b <= a and a == b


def test_in_place_operators():
    group = IntSet([1, 2], width=4)
    group |= [9]
    assert sorted(group) == [1, 2, 9]
    group -= [1]
    assert sorted(group) == [2, 9]
    group &= [9, 12]
    assert sorted(group) == [9]


@pytest.mark.parametrize("other", [5, None, 1.5])
def test_refuses_non_iterables(other):
    for operate in (operator.and_, operator.le, operator.sub):
        with pytest.raises(TypeError):
            operate(IntSet([1], width=2), other)
        with pytest.raises(TypeError):
            operate(other, IntSet([1], width=2))


def test_reflected_operators_are_overridden():
    for name in ("__rand__", "__ror__", "__rsub__", "__rxor__"):
        assert getattr(IntSet, name) is not getattr(AbstractSet, name), name


@pytest.mark.parametrize("left, right", [({1, 2}, [2, 3]), ([0, 7], {7}), (set(), [5])])
def test_reflected_operators_agree_with_builtin_set(left, right):
    group = IntSet(right, width=4)
    assert set(left & group) == set(left) & set(right)
    assert set(left | group) == set(left) | set(right)
    assert set(left - group) == set(left) - set(right)
    assert set(left ^ group) == set(left) ^ set(right)


def test_reflected_operators_dont_enumerate():
    """`Set`'s own reflected operators iterate both sides."""
    for result, size in [
        ({1} | HUGE, 2**64),
        ({1, 2} & HUGE, 2),
        ({1} ^ HUGE, 2**64 - 1),
        (iter([2**64 - 1]) - HUGE, 0),
    ]:
        assert isinstance(result, IntSet)
        assert result.size == size


def test_empty_is_false():
    assert not IntSet()
    assert not IntSet([], width=64)
    assert IntSet([0])


def test_truth_doesnt_use_len():
    """len() overflows past `2**63 - 1`."""
    assert HUGE


@pytest.mark.parametrize("group", [IntSet(), IntSet([1, 2], width=4), HUGE])
def test_pickling_and_copying_round_trip(group):
    for restored in (
        pickle.loads(pickle.dumps(group)),
        copy.deepcopy(group),
        copy.copy(group),
    ):
        assert restored == group
        assert restored.width == group.width
        assert restored.bdd is group.bdd  # re-interned, not duplicated


def test_pickled_leaves_stay_singletons():
    for leaf in (BDD.ACCEPT, BDD.REJECT):
        assert pickle.loads(pickle.dumps(leaf)) is leaf
        assert copy.deepcopy(leaf) is leaf
