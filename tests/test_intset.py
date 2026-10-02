"""`IntSet` cross-checked against the builtin `set`."""

import copy
import operator
import pickle
import random
from collections.abc import MutableSequence, MutableSet, Sequence
from collections.abc import Set as AbstractSet

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bitpattern.bdd import BDD, less_than, node_count
from bitpattern.pattern import Pattern
from bitpattern.sets import IntSet

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
    """A spread of subsets of `range(2 ** width)`, or all of them if that's cheap."""
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
    group = IntSet(members, width=narrow).widen(wide)
    assert group.width == wide
    check(group, members)


@given(members=GROUPS, width=st.integers(min_value=0, max_value=24))
@settings(max_examples=200, deadline=None)
def test_widening_keeps_the_members_at_larger_widths(members, width):
    group = IntSet(members)
    group = group.widen(group.width + width)
    assert set(group) == members
    assert len(group) == len(members)


def test_widening_doesnt_add_members():
    """`{0}` at width 1 is "bit 0 is clear", ie. every even number."""
    assert list(IntSet([0], width=1).widen(4)) == [0]


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


def test_widen_returns_a_new_set():
    group = IntSet([1], width=2)
    wider = group.widen(8)
    assert wider is not group
    assert (group.width, wider.width) == (2, 8)
    assert set(group) == set(wider) == {1}


def test_widen_does_nothing_if_already_wide_enough():
    group = IntSet([1], width=8)
    assert group.widen(4) is group


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
    assert hash(group.widen(group.width + extra)) == hash(group)


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


@pytest.mark.parametrize("width", range(8))
def test_less_than_agrees_with_builtin(width):
    for bound in range(-1, (1 << width) + 2):
        got = list(IntSet.from_bdd(less_than(bound, width), width))
        assert got == [v for v in range(1 << width) if v < bound]


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
    with pytest.raises(ValueError):
        IntSet.range(-1, 5)


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
    """A slice is the members at those positions, and as a set they're in order."""
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
    """`sequence.index(value, *bounds)`, or None if it isn't there."""
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
