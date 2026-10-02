"""Node-level tests: interning, reduction, and the boolean algebra."""

import gc
import weakref

import pytest

from bitpattern.bdd import (
    BDD,
    accepts,
    cofactors,
    count,
    iterate,
    node_count,
    nth,
    size,
)
from bitpattern.intset import IntSet
from bitpattern.pattern import path_count

WIDTH = 3
UNIVERSE = range(1 << WIDTH)
MASKS = range(1 << len(UNIVERSE))

# A spread of masks for the laws with three operands, which are too slow to check
# over every triple.
SAMPLE = [0, 1, 0b10101010, 0b00001111, 0b11001010, 0b01111110, 0b10000001, 255]


def subset(mask):
    return {value for value in UNIVERSE if mask >> value & 1}


def diagram(mask):
    return IntSet(subset(mask), width=WIDTH).bdd


def parity(width):
    """Popcount parity: two nodes per bit, and exponentially many paths."""
    even, odd = BDD.ACCEPT, BDD.REJECT
    for bit in range(width):
        even, odd = BDD(bit, even, odd), BDD(bit, odd, even)
    return even, odd


@pytest.fixture(params=SAMPLE, ids=hex)
def a(request):
    return diagram(request.param)


@pytest.fixture(params=SAMPLE, ids=hex)
def b(request):
    return diagram(request.param)


def test_everything_is_a_bdd():
    for node in (BDD.ACCEPT, BDD.REJECT, BDD(3, BDD.ACCEPT, BDD.REJECT)):
        assert type(node) is BDD


def test_only_reject_is_false():
    assert not BDD.REJECT
    assert BDD.ACCEPT
    assert BDD(3, BDD.ACCEPT, BDD.REJECT) and BDD(3, BDD.REJECT, BDD.ACCEPT)


def test_leaves_cant_be_built():
    with pytest.raises(ValueError):
        BDD(-1, BDD.ACCEPT, BDD.REJECT)


def test_structural_equality_is_identity():
    assert BDD(3, BDD.ACCEPT, BDD.REJECT) is BDD(3, BDD.ACCEPT, BDD.REJECT)
    assert BDD(3, BDD.ACCEPT, BDD.REJECT) is not BDD(3, BDD.REJECT, BDD.ACCEPT)
    assert BDD(3, BDD.ACCEPT, BDD.REJECT) is not BDD(2, BDD.ACCEPT, BDD.REJECT)


def test_equal_sets_share_one_diagram():
    assert IntSet([1, 2, 3], width=WIDTH).bdd is IntSet([3, 2, 1, 3], width=WIDTH).bdd


def test_identical_branches_collapse():
    inner = BDD(3, BDD.ACCEPT, BDD.REJECT)
    assert BDD(5, inner, inner) is inner
    assert BDD(5, BDD.ACCEPT, BDD.ACCEPT) is BDD.ACCEPT


def test_collapsing_leaves_the_branch_alone():
    """`BDD(5, inner, inner)` returns `inner`, which mustn't get reinitialized."""
    inner = BDD(3, BDD.ACCEPT, BDD.REJECT)
    BDD(5, inner, inner)
    assert (inner.bit, inner.left, inner.right) == (3, BDD.ACCEPT, BDD.REJECT)


def test_reinterning_leaves_the_node_alone():
    original = BDD(3, BDD.ACCEPT, BDD.REJECT)
    BDD(3, BDD.ACCEPT, BDD.REJECT)
    assert (original.bit, original.left, original.right) == (3, BDD.ACCEPT, BDD.REJECT)


def test_children_must_test_lower_bits():
    inner = BDD(3, BDD.ACCEPT, BDD.REJECT)
    with pytest.raises(ValueError):
        BDD(1, inner, BDD.REJECT)
    with pytest.raises(ValueError):
        BDD(3, inner, BDD.REJECT)


def test_collapsing_skips_the_ordering_check():
    """`BDD(b, x, x)` doesn't build a node, so `b` doesn't have to be above `x`."""
    inner = BDD(3, BDD.ACCEPT, BDD.REJECT)
    assert BDD(1, inner, inner) is inner


def test_leaves_are_below_every_bit():
    assert BDD.ACCEPT.bit == BDD.REJECT.bit == -1


def test_leaves_are_their_own_branches():
    for leaf in (BDD.ACCEPT, BDD.REJECT):
        assert leaf.left is leaf.right is leaf
        assert cofactors(leaf, 0) == (leaf, leaf)


def test_cofactors_split_only_the_bit_a_node_tests():
    inner = BDD(3, BDD.ACCEPT, BDD.REJECT)
    assert cofactors(inner, 3) == (BDD.ACCEPT, BDD.REJECT)
    assert cofactors(inner, 5) == (inner, inner)


@pytest.mark.parametrize(
    "node", [BDD.ACCEPT, BDD.REJECT, BDD(3, BDD.ACCEPT, BDD.REJECT)]
)
@pytest.mark.parametrize("other", [5, "x", None, 1.5])
def test_operators_refuse_anything_but_nodes(node, other):
    """They return NotImplemented before the cache, which couldn't key on these."""
    for operate in (lambda: node & other, lambda: node | other, lambda: node - other):
        with pytest.raises(TypeError, match="unsupported operand"):
            operate()


def test_double_negation(a):
    assert ~~a is a


def test_idempotence(a):
    assert a & a is a
    assert a | a is a


def test_complement(a):
    assert a & ~a is BDD.REJECT
    assert a | ~a is BDD.ACCEPT


def test_identity_elements(a):
    assert a & BDD.ACCEPT is a
    assert a | BDD.REJECT is a
    assert a & BDD.REJECT is BDD.REJECT
    assert a | BDD.ACCEPT is BDD.ACCEPT


def test_commutativity(a, b):
    assert a & b is b & a
    assert a | b is b | a


def test_de_morgan(a, b):
    assert ~(a & b) is ~a | ~b
    assert ~(a | b) is ~a & ~b


def test_absorption(a, b):
    assert a & (a | b) is a
    assert a | (a & b) is a


def test_difference_is_conjunction_with_complement(a, b):
    assert a - b is a & ~b


@pytest.mark.parametrize("third", SAMPLE, ids=hex)
def test_associativity(a, b, third):
    c = diagram(third)
    assert (a & b) & c is a & (b & c)
    assert (a | b) | c is a | (b | c)


@pytest.mark.parametrize("third", SAMPLE, ids=hex)
def test_distributivity(a, b, third):
    c = diagram(third)
    assert a & (b | c) is (a & b) | (a & c)
    assert a | (b & c) is (a | b) & (a | c)


def test_accept_is_everything():
    assert count(BDD.ACCEPT, 0) == 1
    assert count(BDD.ACCEPT, 4) == 16
    assert list(iterate(BDD.ACCEPT, 4)) == list(range(16))


def test_reject_is_nothing():
    assert count(BDD.REJECT, 0) == 0
    assert count(BDD.REJECT, 8) == 0
    assert list(iterate(BDD.REJECT, 8)) == []


def test_free_high_bits_multiply_the_count():
    evens = BDD(0, BDD.ACCEPT, BDD.REJECT)
    assert count(evens, 1) == 1
    assert count(evens, 4) == 8
    assert list(iterate(evens, 4)) == [0, 2, 4, 6, 8, 10, 12, 14]
    assert accepts(evens, 2**100) and not accepts(evens, 2**100 + 1)


def test_nth_agrees_with_iterate():
    evens = BDD(0, BDD.ACCEPT, BDD.REJECT)
    members = list(iterate(evens, 4))
    assert [nth(evens, 4, i) for i in range(len(members))] == members
    assert [nth(evens, 4, ~i) for i in range(len(members))] == members[::-1]


def test_nth_out_of_range():
    for item in (8, -9, 100):
        with pytest.raises(IndexError):
            nth(BDD(0, BDD.ACCEPT, BDD.REJECT), 4, item)


def test_zero_width_holds_only_zero():
    assert list(iterate(BDD.ACCEPT, 0)) == [0]
    assert nth(BDD.ACCEPT, 0, 0) == 0


def test_width_must_cover_the_diagram():
    with pytest.raises(ValueError):
        count(BDD(5, BDD.ACCEPT, BDD.REJECT), 5)


@pytest.mark.parametrize("mask", MASKS)
def test_matches_the_subset_it_was_built_from(mask):
    node, members = diagram(mask), sorted(subset(mask))
    assert list(iterate(node, WIDTH)) == members
    assert count(node, WIDTH) == len(members)
    assert [value for value in UNIVERSE if accepts(node, value)] == members


@pytest.mark.parametrize(
    "node", [BDD.ACCEPT, BDD.REJECT, BDD(3, BDD.ACCEPT, BDD.REJECT)]
)
def test_nodes_are_not_sequences(node):
    """Only a width says which integers a diagram is about, so nodes can't count."""
    for name in ("__len__", "__iter__", "__getitem__"):
        assert not hasattr(node, name), name


@pytest.mark.parametrize(
    "operate",
    [
        lambda x: ~x,
        lambda x: x & x,
        lambda x: x & BDD.ACCEPT,
        lambda x: x | BDD.REJECT,
        lambda x: x - BDD.ACCEPT,
        size,
        node_count,
        path_count,
    ],
    ids=[
        "invert",
        "and-self",
        "and-accept",
        "or-reject",
        "sub",
        "size",
        "node_count",
        "path_count",
    ],
)
def test_memoising_doesnt_keep_nodes_alive(operate):
    """Memo tables hold nodes weakly. `x & BDD.ACCEPT` is `x`, so the results have to
    be held weakly too, or they'd keep their own keys alive."""

    def compute():
        node = BDD(
            1001, BDD(1000, BDD.ACCEPT, BDD.REJECT), BDD.REJECT
        )  # bits nothing else uses
        operate(node)
        return weakref.ref(node)

    probe = compute()
    gc.collect()
    assert probe() is None


def test_results_stay_memoised_while_alive():
    node = BDD(1001, BDD(1000, BDD.ACCEPT, BDD.REJECT), BDD.REJECT)
    assert ~node is ~node
    assert node & ~node is BDD.REJECT


def test_shared_diagrams_stay_polynomial():
    """Without memoising, these would walk all 2**64 paths."""
    even, odd = parity(64)
    assert ~even is odd
    assert even & odd is BDD.REJECT
    assert even | odd is BDD.ACCEPT
    assert size(even) == 2**63
    assert path_count(even) == 2**63
