"""Node-level tests: interning, reduction, and the boolean algebra."""

import pytest

from bitpattern.bdd import ACCEPT, BDD, BDDLeaf, REJECT, count, index, iterate
from bitpattern.sets import IntSet

WIDTH = 3
UNIVERSE = range(1 << WIDTH)
MASKS = range(1 << len(UNIVERSE))

# A handful of masks spread across the lattice, for the laws that need three
# operands and so cannot be checked over every triple.
SAMPLE = [0, 1, 0b10101010, 0b00001111, 0b11001010, 0b01111110, 0b10000001, 255]


def subset(mask):
    return {value for value in UNIVERSE if mask >> value & 1}


def node(mask):
    return IntSet(subset(mask), WIDTH).bdd


@pytest.fixture(params=SAMPLE, ids=hex)
def a(request):
    return node(request.param)


@pytest.fixture(params=SAMPLE, ids=hex)
def b(request):
    return node(request.param)


class TestInterning:
    def test_leaves_are_singletons(self):
        assert BDDLeaf(True) is ACCEPT
        assert BDDLeaf(False) is REJECT
        assert BDDLeaf(1) is ACCEPT and BDDLeaf(0) is REJECT

    def test_structural_equality_is_identity(self):
        assert BDD(3, ACCEPT, REJECT) is BDD(3, ACCEPT, REJECT)
        assert BDD(3, ACCEPT, REJECT) is not BDD(3, REJECT, ACCEPT)
        assert BDD(3, ACCEPT, REJECT) is not BDD(2, ACCEPT, REJECT)

    def test_equal_sets_share_one_diagram(self):
        assert IntSet([1, 2, 3], WIDTH).bdd is IntSet([3, 2, 1, 3], WIDTH).bdd

    def test_leaf_and_node_interns_are_separate(self):
        assert BDDLeaf.intern is not BDD.intern

    def test_identical_branches_collapse(self):
        inner = BDD(3, ACCEPT, REJECT)
        assert BDD(5, inner, inner) is inner
        assert BDD(5, ACCEPT, ACCEPT) is ACCEPT

    def test_collapse_does_not_mutate_the_shared_branch(self):
        """Regression: __new__ hands back a node that __init__ would overwrite."""
        inner = BDD(3, ACCEPT, REJECT)
        before = (inner.bit, inner.left, inner.right)
        BDD(5, inner, inner)
        assert (inner.bit, inner.left, inner.right) == before

    def test_reconstruction_does_not_mutate(self):
        original = BDD(3, ACCEPT, REJECT)
        BDD(3, ACCEPT, REJECT)
        assert (original.bit, original.left, original.right) == (3, ACCEPT, REJECT)


class TestOrdering:
    def test_children_must_test_lower_bits(self):
        inner = BDD(3, ACCEPT, REJECT)
        with pytest.raises(ValueError):
            BDD(1, inner, REJECT)
        with pytest.raises(ValueError):
            BDD(3, inner, REJECT)

    def test_collapse_wins_over_the_ordering_check(self):
        """`BDD(b, x, x)` builds no node, so `b` never has to dominate `x`."""
        inner = BDD(3, ACCEPT, REJECT)
        assert BDD(1, inner, inner) is inner


class TestOperandGuards:
    """Operators decline non-nodes rather than reaching for `.bit` or `.value`."""

    @pytest.mark.parametrize("node", [ACCEPT, REJECT, BDD(3, ACCEPT, REJECT)])
    @pytest.mark.parametrize("other", [5, "x", None, 1.5])
    def test_binary_operators_reject_foreign_operands(self, node, other):
        for operate in (lambda: node & other, lambda: node | other, lambda: node - other):
            with pytest.raises(TypeError):
                operate()


class TestAlgebra:
    def test_double_negation(self, a):
        assert ~~a is a

    def test_idempotence(self, a):
        assert a & a is a
        assert a | a is a

    def test_complement(self, a):
        assert a & ~a is REJECT
        assert a | ~a is ACCEPT

    def test_identity_elements(self, a):
        assert a & ACCEPT is a
        assert a | REJECT is a
        assert a & REJECT is REJECT
        assert a | ACCEPT is ACCEPT

    def test_commutativity(self, a, b):
        assert a & b is b & a
        assert a | b is b | a

    def test_de_morgan(self, a, b):
        assert ~(a & b) is ~a | ~b
        assert ~(a | b) is ~a & ~b

    def test_absorption(self, a, b):
        assert a & (a | b) is a
        assert a | (a & b) is a

    def test_difference_is_conjunction_with_complement(self, a, b):
        assert a - b is a & ~b

    @pytest.mark.parametrize("third", SAMPLE, ids=hex)
    def test_associativity(self, a, b, third):
        c = node(third)
        assert (a & b) & c is a & (b & c)
        assert (a | b) | c is a | (b | c)

    @pytest.mark.parametrize("third", SAMPLE, ids=hex)
    def test_distributivity(self, a, b, third):
        c = node(third)
        assert a & (b | c) is (a & b) | (a & c)
        assert a | (b & c) is (a | b) & (a | c)


class TestScoping:
    """`count`/`index`/`iterate` read a predicate over a chosen bit range."""

    def test_accept_fills_its_scope(self):
        assert count(ACCEPT, -1) == 1
        assert count(ACCEPT, 3) == 16
        assert list(iterate(ACCEPT, 3)) == list(range(16))

    def test_reject_is_empty_at_every_scope(self):
        assert count(REJECT, -1) == 0
        assert count(REJECT, 7) == 0
        assert list(iterate(REJECT, 7)) == []

    def test_free_high_bits_multiply_the_count(self):
        evens = BDD(0, ACCEPT, REJECT)
        assert count(evens, 0) == 1
        assert count(evens, 3) == 8
        assert list(iterate(evens, 3)) == [0, 2, 4, 6, 8, 10, 12, 14]

    def test_index_agrees_with_iterate(self):
        evens = BDD(0, ACCEPT, REJECT)
        members = list(iterate(evens, 3))
        assert [index(evens, 3, i) for i in range(len(members))] == members
        assert [index(evens, 3, ~i) for i in range(len(members))] == members[::-1]

    def test_index_rejects_out_of_range(self):
        for item in (8, -9, 100):
            with pytest.raises(IndexError):
                index(BDD(0, ACCEPT, REJECT), 3, item)

    def test_empty_scope_holds_only_zero(self):
        assert list(iterate(ACCEPT, -1)) == [0]
        assert index(ACCEPT, -1, 0) == 0

    def test_scope_must_cover_the_diagram(self):
        with pytest.raises(ValueError):
            count(BDD(5, ACCEPT, REJECT), 3)


class TestNodeSequence:
    """A node read over its own bit range, which is what `__len__`/`[]` do."""

    @pytest.mark.parametrize("mask", MASKS)
    def test_matches_the_subset_it_was_built_from(self, mask):
        members = sorted(subset(mask))
        diagram = node(mask)
        assert list(iterate(diagram, WIDTH - 1)) == members
        assert count(diagram, WIDTH - 1) == len(members)
