"""Tuple codecs, with their products and projections checked against brute force."""

import itertools
from functools import cached_property
from ipaddress import IPv4Address
from typing import assert_type

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern.bdd import node_count
from bitpattern.codecs import (
    BDDSet,
    Codec,
    Tuple,
    float16,
    int8,
    ipv4,
    ipv6,
    product,
    project,
    uint8,
    uint16,
)

PAIR = Tuple(uint8, int8)
LEFTS = [uint8.range(10, 20), uint8.set([0, 255]), uint8.none, uint8.all]
RIGHTS = [int8.range(-5, 5), int8.pattern("1*.*"), int8.none, int8.all]


class Small(Codec[int]):
    """0 to 2, which leaves one of its 4 bit patterns unused."""

    def encode(self, value: object) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{value!r} isn't an int")
        if not 0 <= value <= 2:
            raise ValueError(f"{value!r} isn't 0, 1 or 2")
        return value

    def decode(self, bits: int) -> int:
        return bits

    @cached_property
    def all(self) -> BDDSet[int]:
        return self.set(range(3))


def test_tuples_know_their_field_types():
    sockets = Tuple(ipv4, uint16)
    assert_type(sockets, Tuple[IPv4Address, int])
    address, port = sockets.decode(sockets.encode((IPv4Address("10.0.0.1"), 80)))
    assert_type(address, IPv4Address)
    assert_type(port, int)
    assert (address, port) == (IPv4Address("10.0.0.1"), 80)


def test_products_know_their_types():
    pairs = ipv4.all * uint16.all
    assert_type(pairs, BDDSet[tuple[IPv4Address, int]])
    assert_type(pairs * float16.all, BDDSet[tuple[tuple[IPv4Address, int], float]])
    triples = product(ipv4.all, uint16.all, float16.all)
    assert_type(triples, BDDSet[tuple[IPv4Address, int, float]])
    assert_type(project(triples, 0), BDDSet[IPv4Address])
    assert_type(project(triples, 1), BDDSet[int])
    assert_type(project(triples, -1), BDDSet[float])


def test_the_first_field_is_on_top():
    assert PAIR.width == 16
    assert PAIR.encode((1, -1)) == 0x01FF
    assert PAIR.decode(0x01FF) == (1, -1)


@pytest.mark.parametrize("left", LEFTS)
@pytest.mark.parametrize("right", RIGHTS)
def test_products_hold_every_pair_in_order(left, right):
    pairs = list(itertools.product(left, right))
    assert (left * right).size == len(pairs)
    assert list(left * right) == sorted(pairs, key=PAIR.encode)
    assert left * right == product(left, right)


def test_products_cost_their_fields_nodes():
    """However many tuples there are."""
    spare = ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")
    ports = uint16.range(1024, 65536)
    sockets = spare * ports
    assert sockets.size == spare.size * ports.size
    assert node_count(sockets.storage.bdd) == (
        node_count(spare.storage.bdd) + node_count(ports.storage.bdd)
    )


def test_stars_nest_but_product_is_flat():
    """Like itertools.product, taken two at a time or all at once."""
    ones, twos, threes = uint8.set([1]), uint8.set([2]), int8.set([3])
    assert list(ones * twos * threes) == [((1, 2), 3)]
    assert list(product(ones, twos, threes)) == [(1, 2, 3)]


def test_the_product_of_nothing_is_the_empty_tuple():
    assert list(product()) == list(itertools.product()) == [()]


@pytest.mark.parametrize("left", LEFTS)
@pytest.mark.parametrize("right", RIGHTS)
def test_project_undoes_product(left, right):
    pairs = left * right
    assert project(pairs, 0) == (left if right else uint8.none)
    assert project(pairs, 1) == (right if left else int8.none)


@given(
    pairs=st.sets(st.tuples(st.integers(0, 255), st.integers(-128, 127)), max_size=40)
)
@settings(deadline=None)
def test_project_matches_brute_force(pairs):
    source = PAIR.set(pairs)
    assert set(project(source, 0)) == {left for left, _ in pairs}
    assert set(project(source, 1)) == {right for _, right in pairs}


def test_project_counts_back_from_the_end():
    source = product(uint8.set([1]), uint8.set([2]), int8.set([3]))
    assert project(source, -1) == int8.set([3])
    assert project(source, -3) == uint8.set([1])
    with pytest.raises(IndexError):
        project(source, 3)


def test_all_is_every_tuple_of_the_fields_values():
    small = Tuple(Small(2, "small"), Small(2, "small"))
    assert small.all.size == 9
    assert ~small.set([(0, 0)]) == small.all - small.set([(0, 0)])
    assert (3, 0) not in small.all


@pytest.mark.parametrize("value", [(1,), (1, 2, 3), (256, 0), (0, 128)])
def test_tuples_refuse_what_they_cant_encode(value):
    with pytest.raises(ValueError):
        PAIR.encode(value)
    assert value not in PAIR.all


@pytest.mark.parametrize("value", [[1, 2], (0, 1.5), "ab", None])
def test_tuples_refuse_other_types(value):
    with pytest.raises(TypeError):
        PAIR.encode(value)
    assert value not in PAIR.all


@pytest.mark.parametrize("other", [3, {1}, uint8.all.storage])
def test_star_only_takes_bdd_sets(other):
    with pytest.raises(TypeError):
        uint8.all * other


@pytest.mark.parametrize("source", [uint8.all, uint8.all.storage])
def test_project_needs_a_set_of_tuples(source):
    with pytest.raises((TypeError, AttributeError)):
        project(source, 0)


def test_tuples_nest():
    nested = Tuple(Tuple(uint8, uint8), int8)
    assert nested.width == 24
    assert nested.decode(nested.encode(((1, 2), -3))) == ((1, 2), -3)
    assert project(nested.all, 0) == Tuple(uint8, uint8).all


def test_patterns_span_the_fields():
    assert PAIR.pattern("0000.0001.*.*") == uint8.set([1]) * int8.all


def test_tuples_compare_by_their_fields():
    assert Tuple(uint8, int8) == PAIR and hash(Tuple(uint8, int8)) == hash(PAIR)
    assert Tuple(int8, uint8) != PAIR
    assert repr(PAIR) == "tuple[uint8, int8]"
    assert uint8.all * int8.all | PAIR.none == PAIR.all


def test_fields_can_be_any_codec():
    source = float16.range(1.0, 2.0) * uint8.set([7])
    assert (1.5, 7) in source
    assert (2.0, 7) not in source and (1.5, 8) not in source


def test_products_wider_than_128_bits_index_by_position():
    pairs = ipv6.all * uint16.set([80, 443])
    n = 2**126 + 1
    assert pairs[n] == (ipv6.decode(n // 2), 443)
    assert pairs.index(pairs[n]) == n


class Stopping(Codec[int]):
    """A broken codec, whose encode and decode raise StopIteration."""

    def encode(self, value: object) -> int:
        raise StopIteration

    def decode(self, bits: int) -> int:
        raise StopIteration


def test_a_field_raising_stop_iteration_is_a_runtime_error():
    """As is, it would quietly end an iteration, eg. one over the values to encode."""
    pair = Tuple(uint8, Stopping(8, "stopping"))
    with pytest.raises(RuntimeError):
        pair.decode(5)
    with pytest.raises(RuntimeError):
        pair.encode((1, 2))
    with pytest.raises(RuntimeError):
        pair.set([(1, 2), (3, 4)])
