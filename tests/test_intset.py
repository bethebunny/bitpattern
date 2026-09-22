"""`IntSet` cross-checked against the builtin `set`."""

import random
from collections.abc import MutableSet, Set as AbstractSet

import pytest
from hypothesis import assume, given, settings, strategies as st

from bitpattern.bdd import ACCEPT, BDD, REJECT, less_than
from bitpattern.pattern import Pattern
from bitpattern.sets import IntSet

WIDTH = 3
UNIVERSE = range(1 << WIDTH)
MASKS = range(1 << len(UNIVERSE))


def subset(mask):
    return {value for value in UNIVERSE if mask >> value & 1}


VALUES = st.integers(min_value=0, max_value=(1 << 16) - 1)
GROUPS = st.sets(VALUES, max_size=40)

SUBSETS = [subset(mask) for mask in MASKS]
SETS = [IntSet(members, WIDTH) for members in SUBSETS]
NODES = [group.bdd for group in SETS]


def nodes(node) -> int:
    seen, stack = set(), [node]
    while stack:
        current = stack.pop()
        if isinstance(current, BDD) and id(current) not in seen:
            seen.add(id(current))
            stack += [current.left, current.right]
    return len(seen)


def samples(width: int, count: int = 12) -> list[set[int]]:
    """A spread of subsets of `[0, 2 ** width)`; exhaustive while that is cheap."""
    size = 1 << width
    if size <= 4:
        return [{v for v in range(size) if mask >> v & 1} for mask in range(1 << size)]
    chooser = random.Random(width)
    edges = [set(), set(range(size)), {0}, {size - 1}, {0, size - 1}]
    drawn = [
        set(chooser.sample(range(size), chooser.randrange(size)))
        for _ in range(count)
    ]
    return edges + drawn


def check(group: IntSet, members: set[int]):
    """Every read on `group` must agree with the plain set `members`."""
    expected = sorted(members)
    assert len(group) == len(expected)
    assert list(group) == expected
    assert list(group) == sorted(group)
    for value in range(1 << group.width):
        assert (value in group) == (value in members)
    for at, value in enumerate(expected):
        assert group[at] == value
        assert group[at - len(expected)] == value
    for at in (len(expected), -len(expected) - 1):
        with pytest.raises(IndexError):
            group[at]


class TestExhaustive:
    """Every subset of a three-bit universe, read every way."""

    @pytest.mark.parametrize("mask", MASKS)
    def test_reads_agree_with_builtin_set(self, mask):
        check(SETS[mask], SUBSETS[mask])

    @pytest.mark.parametrize("left", MASKS)
    def test_operators_agree_with_builtin_set(self, left):
        """All 256x256 pairs. Diagrams are canonical, so identity is equality."""
        for right in MASKS:
            assert (SETS[left] & SETS[right]).bdd is NODES[left & right]
            assert (SETS[left] | SETS[right]).bdd is NODES[left | right]
            assert (SETS[left] - SETS[right]).bdd is NODES[left & ~right & 0xFF]

    @pytest.mark.parametrize("left", MASKS)
    def test_equality_matches_membership(self, left):
        for right in MASKS:
            assert (SETS[left] == SETS[right]) == (left == right)


class TestMixedWidths:
    """Widening has to restate the bound that lives in `width`, not the diagram."""

    @pytest.mark.parametrize("narrow", [0, 1, 2, 3])
    @pytest.mark.parametrize("wide", [4, 5, 8])
    @pytest.mark.parametrize("members", [set(), {0}, {1}, {0, 1}, {0, 3}, {2, 3}])
    def test_widening_preserves_members(self, narrow, wide, members):
        members = {value for value in members if value.bit_length() <= narrow}
        group = IntSet(members, narrow).widen(wide)
        assert group.width == wide
        check(group, members)

    @pytest.mark.parametrize("left_width", [0, 1, 2, 3, 5])
    @pytest.mark.parametrize("right_width", [0, 1, 2, 3, 5])
    def test_operators_across_widths(self, left_width, right_width):
        width = max(left_width, right_width)
        for left in samples(left_width):
            a = IntSet(left, left_width)
            for right in samples(right_width):
                b = IntSet(right, right_width)
                for got, want in [
                    (a & b, left & right),
                    (a | b, left | right),
                    (a - b, left - right),
                ]:
                    assert got.width == width
                    assert set(got) == want
                    assert len(got) == len(want)

    def test_widening_does_not_invent_members(self):
        """`{0}` at width 1 is "bit 0 clear" -- every even number, read wider."""
        assert list(IntSet([0], 1).widen(4)) == [0]


class TestImmutability:
    """Sets are values: rebind rather than mutate, and they hash."""

    def test_has_no_mutators(self):
        for name in ("add", "discard", "clear", "pop", "remove"):
            assert not hasattr(IntSet(), name), name

    def test_widen_returns_a_new_set(self):
        group = IntSet([1], 2)
        wider = group.widen(8)
        assert wider is not group
        assert (group.width, wider.width) == (2, 8)
        assert set(group) == set(wider) == {1}

    def test_widen_is_a_no_op_when_already_wide_enough(self):
        group = IntSet([1], 8)
        assert group.widen(4) is group

    def test_rebinding_leaves_the_original_alone(self):
        group = IntSet([1, 2], 4)
        other = group
        group |= [9]
        assert sorted(group) == [1, 2, 9]
        assert sorted(other) == [1, 2]

    def test_rejects_negatives(self):
        with pytest.raises(ValueError):
            IntSet([-1])

    def test_negatives_are_never_members(self):
        assert -1 not in IntSet([1], 4)

    def test_non_integers_are_never_members(self):
        assert "1" not in IntSet([1], 4)
        assert None not in IntSet([1], 4)


class TestHashing:
    """Equality is by members, so the hash has to ignore width too."""

    @pytest.mark.parametrize("mask", MASKS)
    def test_equal_sets_hash_equally_across_widths(self, mask):
        forms = [IntSet(SUBSETS[mask], width) for width in range(3, 10)]
        assert len({hash(form) for form in forms}) == 1
        assert all(form == forms[0] for form in forms)

    def test_distinct_sets_hash_apart(self):
        assert len({hash(group) for group in SETS}) == len(MASKS)

    def test_usable_as_dict_keys(self):
        table = {IntSet([1], 1): "a", IntSet([1, 2], 4): "b"}
        assert table[IntSet([1], 8)] == "a"
        assert table[IntSet([1, 2], 2)] == "b"

    def test_empty_and_zero_hash_apart(self):
        assert hash(IntSet()) != hash(IntSet([0]))
        assert IntSet() != IntSet([0])

    def test_hashing_a_huge_set_is_cheap(self):
        """`Set._hash` would iterate 2**64 members; this must not."""
        assert isinstance(hash(IntSet.range(0, 1 << 64, width=64)), int)

    @given(members=GROUPS, extra=st.integers(0, 40))
    @settings(max_examples=200, deadline=None)
    def test_widening_preserves_the_hash(self, members, extra):
        group = IntSet(members)
        assert hash(group.widen(group.width + extra)) == hash(group)


class TestConstruction:
    def test_from_bdd_returns_the_set(self):
        group = IntSet.from_bdd(BDD(0, ACCEPT, REJECT), 3)
        assert isinstance(group, IntSet)
        assert list(group) == [0, 2, 4, 6]

    def test_from_bdd_rejects_a_diagram_wider_than_its_width(self):
        with pytest.raises(ValueError):
            IntSet.from_bdd(BDD(5, ACCEPT, REJECT), 3)

    def test_empty(self):
        group = IntSet()
        assert len(group) == 0 and list(group) == [] and 0 not in group

    def test_width_zero_holds_only_zero(self):
        assert list(IntSet.from_bdd(ACCEPT, 0)) == [0]
        assert list(IntSet.from_bdd(REJECT, 0)) == []

    def test_operators_accept_any_iterable(self):
        group = IntSet([1, 2, 3], 2)
        assert set(group & [2, 3, 9]) == {2, 3}
        assert set(group | iter([9])) == {1, 2, 3, 9}
        assert set(group - (2, 3)) == {1}

    def test_repr_round_trips_the_members(self):
        assert repr(IntSet([3, 1], 2)) == "IntSet([1, 3], width=2)"


class TestProperties:
    values, groups = VALUES, GROUPS

    @given(members=groups)
    @settings(max_examples=200, deadline=None)
    def test_reads_agree_with_builtin_set(self, members):
        group = IntSet(members)
        assert len(group) == len(members)
        assert list(group) == sorted(members)
        for at, value in enumerate(sorted(members)):
            assert group[at] == value
            assert value in group

    @given(left=groups, right=groups)
    @settings(max_examples=200, deadline=None)
    def test_operators_agree_with_builtin_set(self, left, right):
        a, b = IntSet(left), IntSet(right)
        assert set(a & b) == left & right
        assert set(a | b) == left | right
        assert set(a - b) == left - right
        assert (a == b) == (left == right)

    @given(members=groups, extra=values)
    @settings(max_examples=200, deadline=None)
    def test_union_then_difference_restores(self, members, extra):
        assume(extra not in members)
        group = IntSet(members)
        assert set(group | {extra}) == members | {extra}
        assert set((group | {extra}) - {extra}) == members

    @given(members=groups, width=st.integers(min_value=0, max_value=24))
    @settings(max_examples=200, deadline=None)
    def test_widening_preserves_members(self, members, width):
        group = IntSet(members)
        group = group.widen(group.width + width)
        assert set(group) == members
        assert len(group) == len(members)

    @given(members=st.sets(values, min_size=1, max_size=40))
    @settings(max_examples=200, deadline=None)
    def test_index_is_the_inverse_of_iteration(self, members):
        group = IntSet(members)
        ordered = sorted(members)
        for at in range(-len(ordered), len(ordered)):
            assert group[at] == ordered[at]
        with pytest.raises(IndexError):
            group[len(ordered)]


class TestRange:
    """`less_than` and `range`, which cost one node per bit however large."""

    @pytest.mark.parametrize("width", range(8))
    def test_less_than_agrees_with_builtin(self, width):
        for bound in range(-1, (1 << width) + 2):
            got = list(IntSet.from_bdd(less_than(bound, width), width))
            assert got == [v for v in range(1 << width) if v < bound]

    @pytest.mark.parametrize("width", range(6))
    def test_range_agrees_with_builtin(self, width):
        for start in range(1 << width):
            for stop in range(1 << width):
                got = IntSet.range(start, stop, width)
                assert list(got) == list(range(start, max(start, stop)))
                assert got.width == width

    def test_width_defaults_to_fitting_the_stop(self):
        assert IntSet.range(3, 17).width == 5
        assert list(IntSet.range(3, 17)) == list(range(3, 17))
        assert IntSet.range(0, 0).width == 0
        assert list(IntSet.range(5, 2)) == []

    def test_is_linear_in_width_not_in_members(self):
        span = IntSet.range(10**15, 10**18, width=64)
        assert span.size == 10**18 - 10**15
        assert nodes(span.bdd) < 4 * 64

    def test_covers_the_whole_universe(self):
        assert IntSet.range(0, 1 << 64, width=64).bdd is ACCEPT
        assert IntSet.range(0, 10**30, width=64).size == 1 << 64

    def test_rejects_negative_start(self):
        with pytest.raises(ValueError):
            IntSet.range(-1, 5)


class TestSize:
    """`size` is the primitive; `len` is the narrowed view of it."""

    def test_agrees_with_len_when_it_fits(self):
        group = IntSet([1, 2, 3], 8)
        assert group.size == len(group) == 3

    def test_survives_a_universe_len_cannot_express(self):
        universe = IntSet.range(0, 1 << 64, width=64)
        assert universe.size == 2**64
        with pytest.raises(OverflowError):
            len(universe)

    def test_the_boundary(self):
        assert IntSet.range(0, 2**63 - 1, width=64).size == len(
            IntSet.range(0, 2**63 - 1, width=64)
        )
        with pytest.raises(OverflowError):
            len(IntSet.range(0, 2**63, width=64))

    def test_iteration_still_works_on_a_huge_set(self):
        universe = IntSet.range(0, 1 << 64, width=64)
        assert [universe[at] for at in range(3)] == [0, 1, 2]
        assert universe[-1] == 2**64 - 1
        assert universe[2**63] == 2**63


class TestComplement:
    @pytest.mark.parametrize("mask", MASKS)
    def test_agrees_with_builtin_set(self, mask):
        assert set(~SETS[mask]) == set(UNIVERSE) - SUBSETS[mask]

    def test_is_an_involution(self):
        group = IntSet([1, 5], 4)
        assert ~~group == group
        assert (group | ~group).bdd is ACCEPT
        assert (group & ~group).bdd is REJECT

    def test_respects_the_width(self):
        assert list(~IntSet([1], 2)) == [0, 2, 3]
        assert list(~IntSet([1], 3)) == [0, 2, 3, 4, 5, 6, 7]


class TestSample:
    def test_is_always_a_member(self):
        group = IntSet([3, 99, 12345], 16)
        for seed in range(100):
            assert group.sample(random.Random(seed)) in group

    def test_reaches_every_member(self):
        group = IntSet([3, 99, 12345], 16)
        assert {group.sample(random.Random(seed)) for seed in range(60)} == set(group)

    def test_works_on_a_set_larger_than_a_ssize_t(self):
        universe = IntSet.range(0, 1 << 64, width=64)
        assert universe.sample(random.Random(0)) < 2**64

    def test_empty_raises(self):
        with pytest.raises(KeyError):
            IntSet().sample()


class TestSetProtocol:
    """The `MutableSet` surface, with the comparisons answered by the diagram."""

    @pytest.mark.parametrize("left", MASKS)
    def test_comparisons_agree_with_builtin_set(self, left):
        a, x = SETS[left], SUBSETS[left]
        for right in MASKS:
            b, y = SETS[right], SUBSETS[right]
            assert (a <= b) == (x <= y)
            assert (a < b) == (x < y)
            assert (a >= b) == (x >= y)
            assert (a > b) == (x > y)
            assert a.isdisjoint(b) == x.isdisjoint(y)
            assert set(a ^ b) == x ^ y

    def test_comparisons_across_widths(self):
        assert IntSet([1], 1) <= IntSet([1, 3], 2)
        assert not IntSet([1, 3], 2) <= IntSet([1], 1)
        assert IntSet([1], 1).isdisjoint(IntSet([2, 3], 2))

    def test_is_an_immutable_set(self):
        assert isinstance(IntSet(), AbstractSet)
        assert not isinstance(IntSet(), MutableSet)

    def test_comparisons_take_only_intsets(self):
        """Equality with a frozenset would need a frozenset-compatible hash."""
        assert IntSet([1, 2], 4) != frozenset({1, 2})
        assert frozenset({1, 2}) != IntSet([1, 2], 4)
        for compare in (lambda a, b: a <= b, lambda a, b: a < b,
                        lambda a, b: a >= b, lambda a, b: a > b):
            with pytest.raises(TypeError):
                compare(IntSet([1, 2], 4), {1, 2})

    def test_equal_objects_hash_equally(self):
        """The hash contract, checked directly across the types that meet here."""
        values = [IntSet([1, 2], 4), IntSet([1, 2], 8), Pattern("00?1"),
                  frozenset({1, 2}), {1, 2}, frozenset({1, 3}), IntSet([1, 3], 2)]
        for a in values:
            for b in values:
                if a == b and not isinstance(a, set) and not isinstance(b, set):
                    assert hash(a) == hash(b), (a, b)

    def test_antisymmetry(self):
        a, b = IntSet([1, 2], 2), IntSet([1, 2], 8)
        assert a <= b and b <= a and a == b

    def test_in_place_operators(self):
        group = IntSet([1, 2], 4)
        group |= [9]
        assert sorted(group) == [1, 2, 9]
        group -= [1]
        assert sorted(group) == [2, 9]
        group &= [9, 12]
        assert sorted(group) == [9]

    def test_rejects_non_iterables(self):
        for operate in (lambda: IntSet([1], 2) & 5, lambda: IntSet([1], 2) <= 5):
            with pytest.raises(TypeError):
                operate()

    @given(left=GROUPS, right=GROUPS)
    @settings(max_examples=200, deadline=None)
    def test_comparisons_agree_at_larger_widths(self, left, right):
        a, b = IntSet(left), IntSet(right)
        assert (a <= b) == (left <= right)
        assert (a < b) == (left < right)
        assert a.isdisjoint(b) == left.isdisjoint(right)
        assert set(a ^ b) == left ^ right


class TestReflectedOperators:
    """The `Set` mixins' reflected operators iterate both operands."""

    def test_are_not_the_mixins(self):
        for name in ("__rand__", "__ror__", "__rsub__", "__rxor__"):
            assert getattr(IntSet, name) is not getattr(AbstractSet, name), name

    @pytest.mark.parametrize("left, right", [({1, 2}, [2, 3]), ([0, 7], {7}), (set(), [5])])
    def test_agree_with_builtin_set(self, left, right):
        group = IntSet(right, 4)
        assert set(left & group) == set(left) & set(right)
        assert set(left | group) == set(left) | set(right)
        assert set(left - group) == set(left) - set(right)
        assert set(left ^ group) == set(left) ^ set(right)

    def test_do_not_enumerate(self):
        """`{1} | huge` used to iterate all 2**64 members."""
        huge = IntSet.range(0, 1 << 64, width=64)
        assert ({1} | huge).size == 2**64
        assert sorted({1, 2} & huge) == [1, 2]
        assert ({1} ^ huge).size == 2**64 - 1
        assert (iter([2**64 - 1]) - huge).size == 0

    def test_rejects_non_iterables(self):
        with pytest.raises(TypeError):
            5 - IntSet([1], 4)


class TestTruth:
    def test_empty_is_false(self):
        assert not IntSet()
        assert not IntSet([], 64)
        assert IntSet([0])

    def test_does_not_go_through_len(self):
        """`bool` used to call `__len__`, which overflows past 2**63 - 1."""
        assert IntSet.range(0, 1 << 64, width=64)


class TestPickling:
    @pytest.mark.parametrize(
        "group", [IntSet(), IntSet([1, 2], 4), IntSet.range(0, 1 << 64, width=64)]
    )
    def test_round_trips(self, group):
        import copy
        import pickle

        for restored in (pickle.loads(pickle.dumps(group)), copy.deepcopy(group), copy.copy(group)):
            assert restored == group
            assert restored.width == group.width
            assert restored.bdd is group.bdd  # re-interned, not duplicated

    def test_leaves_stay_singletons(self):
        import copy
        import pickle

        for leaf in (ACCEPT, REJECT):
            assert pickle.loads(pickle.dumps(leaf)) is leaf
            assert copy.deepcopy(leaf) is leaf
