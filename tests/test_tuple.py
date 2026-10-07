"""Tuple codecs, with their products and projections checked against brute force."""

import itertools
from functools import cached_property
from ipaddress import IPv4Address
from typing import assert_type

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern.bdd import node_count
from bitpattern.codecs import BDDSet, Codec, Tuple, float16, int8, ipv4, uint8, uint16

PAIR = Tuple(uint8, int8)
LEFTS = [uint8.range(10, 20), uint8.set([0, 255]), uint8.none, uint8.all]
RIGHTS = [int8.range(-5, 5), int8.pattern("1*.*"), int8.none, int8.all]


class Small(Codec[int]):
    """0 to 2, which leaves one of its 4 bit patterns unused."""

    def encode(self, value: object) -> int:
        if isinstance(value, int) and 0 <= value <= 2:
            return value
        raise ValueError(f"{value!r} isn't 0, 1 or 2")

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


def test_the_first_field_is_on_top():
    assert PAIR.width == 16
    assert PAIR.encode((1, -1)) == 0x01FF
    assert PAIR.decode(0x01FF) == (1, -1)


@pytest.mark.parametrize("left", LEFTS)
@pytest.mark.parametrize("right", RIGHTS)
def test_products_hold_every_pair_in_order(left, right):
    pairs = list(itertools.product(left, right))
    product = PAIR.product(left, right)
    assert product.size == len(pairs)
    assert list(product) == sorted(pairs, key=PAIR.encode)


def test_products_cost_their_fields_nodes():
    """However many tuples there are."""
    spare = ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")
    ports = uint16.range(1024, 65536)
    product = Tuple(ipv4, uint16).product(spare, ports)
    assert product.size == spare.size * ports.size
    assert node_count(product.storage.bdd) == (
        node_count(spare.storage.bdd) + node_count(ports.storage.bdd)
    )


@pytest.mark.parametrize("left", LEFTS)
@pytest.mark.parametrize("right", RIGHTS)
def test_project_undoes_product(left, right):
    product = PAIR.product(left, right)
    assert PAIR.project(product, 0) == (left if right else uint8.none)
    assert PAIR.project(product, 1) == (right if left else int8.none)


@given(
    pairs=st.sets(st.tuples(st.integers(0, 255), st.integers(-128, 127)), max_size=40)
)
@settings(deadline=None)
def test_project_matches_brute_force(pairs):
    source = PAIR.set(pairs)
    assert set(PAIR.project(source, 0)) == {left for left, _ in pairs}
    assert set(PAIR.project(source, 1)) == {right for _, right in pairs}


def test_all_is_every_tuple_of_the_fields_values():
    small = Tuple(Small(2, "small"), Small(2, "small"))
    assert small.all.size == 9
    assert ~small.set([(0, 0)]) == small.all - small.set([(0, 0)])
    assert (3, 0) not in small.all


@pytest.mark.parametrize(
    "value", [(1,), (1, 2, 3), [1, 2], (256, 0), (0, 128), (0, 1.5), "ab", None]
)
def test_tuples_refuse_what_they_cant_encode(value):
    with pytest.raises(ValueError):
        PAIR.encode(value)
    assert value not in PAIR.all


@pytest.mark.parametrize(
    "sets", [(uint8.all,), (uint8.all, int8.all, int8.all), (int8.all, int8.all)]
)
def test_product_needs_a_set_for_each_field(sets):
    with pytest.raises(TypeError):
        PAIR.product(*sets)


def test_project_needs_this_tuples_sets():
    with pytest.raises(TypeError):
        PAIR.project(Tuple(int8, uint8).all, 0)


def test_tuples_nest():
    nested = Tuple(Tuple(uint8, uint8), int8)
    assert nested.width == 24
    assert nested.decode(nested.encode(((1, 2), -3))) == ((1, 2), -3)
    assert nested.project(nested.all, 0) == Tuple(uint8, uint8).all


def test_patterns_span_the_fields():
    assert PAIR.pattern("0000.0001.*.*") == PAIR.product(uint8.set([1]), int8.all)


def test_tuples_compare_by_their_fields():
    assert Tuple(uint8, int8) == PAIR and hash(Tuple(uint8, int8)) == hash(PAIR)
    assert Tuple(int8, uint8) != PAIR
    assert repr(PAIR) == "tuple[uint8, int8]"
    assert Tuple(uint8, int8).all | PAIR.none == PAIR.all


def test_fields_can_be_any_codec():
    tagged = Tuple(float16, uint8)
    source = tagged.product(float16.range(1.0, 2.0), uint8.set([7]))
    assert (1.5, 7) in source
    assert (2.0, 7) not in source and (1.5, 8) not in source
