"""Node-level tests: interning, reduction, and the boolean algebra."""

import copy
import functools
import gc
import operator
import pickle
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

from bitpattern import bdd
from bitpattern.bdd import (
    BDD,
    accepts,
    blocks,
    count,
    exactly_these,
    freed_between,
    interval,
    lshift,
    members,
    narrowed,
    node_count,
    nth,
    path_count,
    paths,
    rank,
)
from bitpattern.intset import IntSet

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


@pytest.mark.parametrize(
    "node", [BDD.ACCEPT, BDD.REJECT, BDD(3, BDD.ACCEPT, BDD.REJECT)]
)
@pytest.mark.parametrize("other", [5, "x", None, 1.5])
def test_operators_refuse_anything_but_nodes(node, other):
    """Operators return NotImplemented before reaching the weak cache, which can't
    hold `other`."""
    for operate in (
        lambda: node & other,
        lambda: node | other,
        lambda: node - other,
        lambda: other & node,
        lambda: other | node,
        lambda: other - node,
    ):
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
    assert count(BDD.ACCEPT, width=0) == 1
    assert count(BDD.ACCEPT, width=4) == 16
    assert list(members(BDD.ACCEPT, width=4)) == list(range(16))


def test_reject_is_nothing():
    assert count(BDD.REJECT, width=0) == 0
    assert count(BDD.REJECT, width=8) == 0
    assert list(members(BDD.REJECT, width=8)) == []


def test_free_high_bits_multiply_the_count():
    evens = BDD(0, BDD.ACCEPT, BDD.REJECT)
    assert count(evens, width=1) == 1
    assert count(evens, width=4) == 8
    assert list(members(evens, width=4)) == [0, 2, 4, 6, 8, 10, 12, 14]
    assert accepts(evens, 2**100) and not accepts(evens, 2**100 + 1)


# Python won't write out 10**5000, so it can't be an id as it is.
HUGE = pytest.param(10**5000, id="10**5000")


@pytest.mark.parametrize("width", [-1, 2**31, 2**63, -(2**100), HUGE])
def test_widths_outside_the_range_are_value_errors(width):
    with pytest.raises(ValueError, match="width"):
        count(BDD.ACCEPT, width=width)
    with pytest.raises(ValueError, match="width"):
        members(BDD.REJECT, width=width)  # when called, not when iterated
    with pytest.raises(ValueError, match="width"):
        interval(0, 1, width=width)


@pytest.mark.parametrize("bit", [-1, 2**31 - 1, 2**63, -(2**100), HUGE])
def test_bits_outside_the_range_are_value_errors(bit):
    with pytest.raises(ValueError, match="bit"):
        BDD(bit, BDD.ACCEPT, BDD.REJECT)


@pytest.mark.parametrize("value", [1.5, "3", None])
def test_widths_and_bits_that_arent_ints_are_type_errors(value):
    with pytest.raises(TypeError):
        count(BDD.ACCEPT, width=value)
    with pytest.raises(TypeError):
        BDD(value, BDD.ACCEPT, BDD.REJECT)


class Unreadable:
    """An int-like whose __index__ raises, eg. for a Ctrl-C while it runs."""

    def __init__(self) -> None:
        self.calls = 0

    def __index__(self) -> int:
        self.calls += 1
        raise ZeroDivisionError


@pytest.mark.parametrize(
    "read",
    [
        lambda value: exactly_these([1, value], width=0),
        lambda value: nth(BDD.ACCEPT, value, width=8),
        lambda value: accepts(BDD.ACCEPT, value),
    ],
)
def test_an_error_from_index_comes_out_at_once(read):
    """Only an OverflowError means an int needs reading some slower way. Calling
    __index__ again could lose any other error."""
    value = Unreadable()
    with pytest.raises(ZeroDivisionError):
        read(value)
    assert value.calls == 1


def test_shifting_a_bit_out_of_range_is_a_value_error():
    node = BDD(5, BDD(1, BDD.ACCEPT, BDD.REJECT), BDD.REJECT)
    assert lshift(node, -1) is BDD(4, BDD(0, BDD.ACCEPT, BDD.REJECT), BDD.REJECT)
    for by in (-2, 2**31 - 3, 2**40, -(2**40)):
        with pytest.raises(ValueError, match="shifting"):
            lshift(node, by)
    assert lshift(BDD.ACCEPT, 2**100) is BDD.ACCEPT  # a leaf tests no bits


def test_shifting_by_a_huge_int_names_it_by_its_size():
    """Writing it out in decimal would take seconds."""
    node = BDD(5, BDD.ACCEPT, BDD.REJECT)
    with pytest.raises(ValueError, match="by an int of 4000001 bits moves bit 5 out"):
        lshift(node, 1 << 4_000_000)
    assert lshift(BDD.ACCEPT, -(1 << 4_000_000)) is BDD.ACCEPT


def test_errors_name_ints_past_the_digit_limit_by_their_size():
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        with pytest.raises(ValueError, match="got an int of 2326 bits"):
            exactly_these([-(10**700)], width=8)
        with pytest.raises(ValueError, match="width an int of 2326 bits is out of"):
            count(BDD.ACCEPT, width=10**700)
    finally:
        sys.set_int_max_str_digits(previous)


def test_freed_between_clamps_its_bounds():
    node = BDD(5, BDD(1, BDD.ACCEPT, BDD.REJECT), BDD.REJECT)
    assert freed_between(node, low_bit=-(2**100), high_bit=2**100) is BDD.ACCEPT
    assert freed_between(node, low_bit=2**100, high_bit=2**200) is node


class Index:
    """Not an int, but converts to one losslessly, like numpy's integers."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    def __index__(self) -> int:
        self.calls += 1
        return self.value


@pytest.mark.parametrize("low, high", [(Index(-(2**100)), 8), (Index(2**100), 2**200)])
def test_freed_between_clamps_int_likes_past_an_i64(low, high):
    """Without comparing them, which an int-like might not do."""
    node = BDD(5, BDD(1, BDD.ACCEPT, BDD.REJECT), BDD.REJECT)
    assert freed_between(node, low_bit=low, high_bit=high) is freed_between(
        node, low_bit=operator.index(low), high_bit=high
    )


@pytest.mark.parametrize("member", [Index(2**63), Index(2**64 + 5), Index(2**200)])
def test_exactly_these_reads_an_int_like_once(member):
    """Reading it again could lose an error its __index__ raised, eg. for a Ctrl-C."""
    value = member.value
    assert exactly_these([1, member], width=0) == exactly_these([1, value], width=0)
    assert member.calls == 1


@pytest.mark.parametrize("member", [Index(-3)])
def test_exactly_these_names_a_negative_int_like_by_its_int(member):
    with pytest.raises(ValueError, match="got -3$"):
        exactly_these([member], width=0)


def test_equality_with_anything_else_is_up_to_it():
    assert BDD.ACCEPT == mock.ANY
    assert (BDD.ACCEPT != mock.ANY) is False
    assert BDD.REJECT != 0 and BDD.ACCEPT != 1


def test_nodes_match_by_bit_and_branches():
    match BDD(3, BDD.ACCEPT, BDD.REJECT):
        case BDD(bit, left, right):
            assert (bit, left, right) == (3, BDD.ACCEPT, BDD.REJECT)


def test_copies_are_the_node_itself():
    node = BDD(3, BDD.ACCEPT, BDD.REJECT)
    assert copy.copy(node) is node and copy.deepcopy(node) is node
    wide = IntSet([1, 5], width=400)  # deep enough to overflow a recursive deepcopy
    assert copy.deepcopy(wide) == wide


def test_exactly_these_takes_values_in_any_order():
    assert exactly_these([2, 1, 2], width=2) == exactly_these([1, 2], width=2)
    node, width = exactly_these({2**61, 3}, width=62)
    assert list(members(node, width=width)) == [3, 2**61]
    wide = [2**200, 5, 2**199, 5]
    node, width = exactly_these(wide, width=201)
    assert list(members(node, width=width)) == sorted(set(wide))


@pytest.mark.parametrize(
    "values, width, wide",
    [
        ([], 0, 0),
        ([], 5, 5),
        ([0], 0, 0),
        ([1, 4], 2, 3),
        ([2**127], 0, 128),
        ([2**128, 3], 8, 129),
        ([1, 2**300, 5], 0, 301),
    ],
)
def test_exactly_these_is_as_wide_as_its_values_need(values, width, wide):
    node, got = exactly_these(values, width=width)
    assert got == wide
    assert list(members(node, width=got)) == sorted(values)


def test_a_long_stream_with_repeats_takes_little_room():
    node, width = exactly_these((value % 1000 for value in range(2_000_000)), width=0)
    assert list(members(node, width=width)) == list(range(1000))


@pytest.mark.parametrize("value", [-1, -(2**64), -(2**200)])
def test_exactly_these_refuses_negative_values(value):
    with pytest.raises(ValueError, match="can't hold negative numbers"):
        exactly_these([1, 2**130, value], width=8)


def test_exactly_these_names_a_huge_value_by_its_size():
    """Python won't write out an int this long."""
    with pytest.raises(ValueError, match="got an int of 20001 bits"):
        exactly_these([1, -(2**20_000)], width=8)


def test_bits_past_a_value_are_its_sign():
    high, low = BDD(200, BDD.REJECT, BDD.ACCEPT), BDD(3, BDD.REJECT, BDD.ACCEPT)
    assert accepts(high, -1) and not accepts(high, 5)
    assert accepts(high, 2**200) and not accepts(high, -(2**300))
    assert accepts(low, -8) and not accepts(low, -(2**200))


def test_nth_agrees_with_members():
    evens = BDD(0, BDD.ACCEPT, BDD.REJECT)
    values = list(members(evens, width=4))
    assert [nth(evens, i, width=4) for i in range(len(values))] == values
    assert [nth(evens, ~i, width=4) for i in range(len(values))] == values[::-1]


@pytest.mark.parametrize("mask", MASKS)
def test_rank_counts_the_members_below_a_value(mask):
    node = diagram(mask)
    for width in (WIDTH, WIDTH + 2):
        everything = list(members(node, width=width))
        for value in range(-1, (1 << width) + 2):
            below = sum(member < value for member in everything)
            assert rank(node, value, width=width) == below


@pytest.mark.parametrize("mask", MASKS)
def test_blocks_are_the_aligned_runs_of_members(mask):
    node = diagram(mask)
    for width in (WIDTH, WIDTH + 2):
        found = list(blocks(node, width=width))
        assert [
            value for first, free in found for value in range(first, first + 2**free)
        ] == list(members(node, width=width))
        assert all(first % 2**free == 0 for first, free in found)


@pytest.mark.parametrize("mask", MASKS)
def test_narrowed_keeps_the_members_below_the_width(mask):
    node = diagram(mask)
    for width in range(WIDTH + 1):
        expected = [value for value in subset(mask) if value < 1 << width]
        assert list(members(narrowed(node, width=width), width=width)) == expected


@pytest.mark.parametrize("mask", MASKS)
def test_subset_and_disjoint_agree_with_sets(mask):
    for other in MASKS:
        a, b = diagram(mask), diagram(other)
        assert bdd.subset(a, b) == (subset(mask) <= subset(other))
        assert bdd.disjoint(a, b) == subset(mask).isdisjoint(subset(other))


def test_nth_out_of_range():
    for item in (8, -9, 100):
        with pytest.raises(IndexError):
            nth(BDD(0, BDD.ACCEPT, BDD.REJECT), item, width=4)


def test_zero_width_holds_only_zero():
    assert list(members(BDD.ACCEPT, width=0)) == [0]
    assert nth(BDD.ACCEPT, 0, width=0) == 0


@pytest.mark.parametrize("width", [3, 5])
@pytest.mark.parametrize(
    "call",
    [
        count,
        functools.partial(nth, n=0),
        functools.partial(rank, value=0),
        members,
        blocks,
        paths,
        functools.partial(bdd.tensor, code=1, bits=8, version=None),
    ],
    ids=["count", "nth", "rank", "members", "blocks", "paths", "tensor"],
)
def test_width_must_cover_the_diagram(call, width):
    with pytest.raises(ValueError, match=f"^node tests bit 5, outside width {width}$"):
        call(BDD(5, BDD.ACCEPT, BDD.REJECT), width=width)


@pytest.mark.parametrize("mask", MASKS)
def test_matches_the_subset_it_was_built_from(mask):
    node, expected = diagram(mask), sorted(subset(mask))
    assert list(members(node, width=WIDTH)) == expected
    assert count(node, width=WIDTH) == len(expected)
    assert [value for value in UNIVERSE if accepts(node, value)] == expected


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
        lambda x: x.size,
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
    """Memo tables hold nodes weakly. `x & BDD.ACCEPT` is `x`, so results have to be
    held weakly too. A strong result `x` would keep the key `x` alive."""

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
    assert even.size == 2**63
    assert path_count(even) == 2**63


@pytest.mark.parametrize("mask", MASKS)
def test_lshift_moves_every_bit(mask):
    node = diagram(mask)
    up = lshift(node, 2)
    assert set(members(up, width=WIDTH + 2)) == {
        value << 2 | low for value in subset(mask) for low in range(4)
    }
    assert lshift(up, -2) is node


@pytest.mark.parametrize("low, high", [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)])
@pytest.mark.parametrize("mask", SAMPLE, ids=hex)
def test_freed_between_frees_a_range_of_bits(mask, low, high):
    freed = (1 << high) - (1 << low)
    settings = [bits for bits in UNIVERSE if not bits & ~freed]
    expected = {value & ~freed | bits for value in subset(mask) for bits in settings}
    node = freed_between(diagram(mask), low_bit=low, high_bit=high)
    assert set(members(node, width=WIDTH)) == expected


def test_lshift_moves_each_shared_node_once():
    """Parity has two nodes per bit but 2**64 paths, so the test only finishes if
    lshift visits each node once."""
    even, _ = parity(64)
    assert node_count(lshift(even, 1)) == node_count(even)


def test_freed_between_frees_each_shared_node_once():
    """Parity has two nodes per bit but 2**64 paths, so the test only finishes if
    freed_between visits each node once."""
    even, _ = parity(64)
    assert freed_between(even, low_bit=0, high_bit=1) is BDD.ACCEPT


def test_freeing_nothing_changes_nothing():
    even, _ = parity(8)
    assert freed_between(even, low_bit=3, high_bit=3) is even
    assert freed_between(even, low_bit=5, high_bit=2) is even
    assert lshift(even, 0) is even
    assert freed_between(even, low_bit=0, high_bit=8) is BDD.ACCEPT


def chain(value, width):
    """Exactly `value` over `width` bits, one node per bit, built a node at a time."""
    node = BDD.ACCEPT
    for bit in range(width):
        if value >> bit & 1:
            node = BDD(bit, BDD.REJECT, node)
        else:
            node = BDD(bit, node, BDD.REJECT)
    return node


def on_a_small_stack(run):
    """`run()` in a thread with 512 KB of stack, which walking a diagram 20,000 bits
    deep by recursion would overflow."""
    threading.stack_size(512 * 1024)
    try:
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(run).result()
    finally:
        threading.stack_size(0)


def test_deep_diagrams_dont_overflow_the_stack():
    def run():
        zero, one = chain(0, 20_000), chain(1, 20_000)
        both = zero | one
        assert zero & one is BDD.REJECT
        assert both & zero is zero
        assert zero ^ one is both
        assert both - one is zero
        assert ~~zero is zero
        assert lshift(lshift(zero, 3), -3) is zero
        assert freed_between(zero, low_bit=0, high_bit=1) is both
        assert path_count(both) == 1

    on_a_small_stack(run)


def test_building_deep_diagrams_doesnt_overflow_the_stack():
    def run():
        top = 1 << 19_999
        group = IntSet([0, 5, top], width=20_000)
        assert group.size == 3 and 5 in group and group[-1] == top
        span = IntSet.range(3, top, width=20_000)
        assert span.size == top - 3 and span[0] == 3 and span[-1] == top - 1

    on_a_small_stack(run)


def test_pickling_deep_diagrams_doesnt_recurse():
    def run():
        group = IntSet([1, 2**7999], width=8000)
        assert pickle.loads(pickle.dumps(group)) == group
        assert pickle.loads(pickle.dumps(group.bdd)) is group.bdd

    on_a_small_stack(run)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [(3, 5, 0)],
        [(0, 1, 0), (0, 2, 0)],
        [(0, -1, 1)],
        [(0, 2**40, 1)],
        [(0, 0, 2**32)],
        [(0, 0, 2**70)],
    ],
)
def test_rebuilding_from_rows_that_arent_a_diagram_is_a_value_error(rows):
    with pytest.raises(ValueError):
        bdd._rebuilt(rows)
