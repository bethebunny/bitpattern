"""Hypothesis strategies, end to end. This is what the library was built for."""

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bitpattern import IntSet, Pattern
from bitpattern.codecs import float16, float64, ipv4
from bitpattern.strategies import from_set

SETTINGS = settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

# The motivating example: positive floats, minus infinity.
REALS = float64.positive - float64.infinities


@given(value=from_set(IntSet([3, 99, 12345], width=16)))
@SETTINGS
def test_draws_only_members(value):
    assert value in {3, 99, 12345}


@given(value=from_set(IntSet.range(10, 20)))
@SETTINGS
def test_draws_from_a_range(value):
    assert 10 <= value < 20


@given(value=from_set(Pattern("*1.*.*.0000.1111.?01?")))
@SETTINGS
def test_draws_from_a_pattern(value):
    assert value in Pattern("*1.*.*.0000.1111.?01?")


@given(value=from_set(IntSet.range(0, 1 << 64, width=64)))
@SETTINGS
def test_draws_from_a_set_len_cant_measure(value):
    assert 0 <= value < 2**64


def test_an_empty_set_draws_nothing():
    assert from_set(IntSet()).is_empty
    assert from_set(float64.none).is_empty


@given(members=st.sets(st.integers(0, 255), min_size=1, max_size=20), data=st.data())
@settings(max_examples=100, deadline=None)
def test_draws_from_any_set(members, data):
    assert data.draw(from_set(IntSet(members))) in members


@given(value=from_set(REALS))
@SETTINGS
def test_draws_finite_positive_floats(value):
    assert math.isfinite(value) and value > 0


@given(value=from_set(float64.range(1.0, 2.0)))
@SETTINGS
def test_draws_floats_from_a_range(value):
    assert 1.0 <= value < 2.0


@given(value=from_set(float64.subnormal))
@SETTINGS
def test_draws_subnormals(value):
    assert value != 0 and abs(value) < 2.2250738585072014e-308


@given(value=from_set(float64.nan))
@SETTINGS
def test_draws_nans(value):
    assert math.isnan(value)


@given(value=from_set(float64.finite))
@SETTINGS
def test_draws_from_every_finite_float(value):
    assert math.isfinite(value)


def test_one_sign_fits_in_len_but_both_dont():
    assert REALS.size == 2**63 - 2**52 - 1 == len(REALS)
    assert float64.finite.size == 2**64 - 2**53
    with pytest.raises(OverflowError):
        len(float64.finite)


def test_index_zero_is_the_natural_shrink_target():
    """Shrinking the index shrinks the float, since their orders agree."""
    positives = float64.finite & float64.sign_clear
    assert positives[0] == 0.0
    assert positives[1] == 5e-324


@given(value=st.sampled_from(float16.range(1.0, 1.01)))
@SETTINGS
def test_small_bdd_sets_work_with_sampled_from(value):
    """sampled_from copies its sequence into a tuple, so only small ones."""
    assert isinstance(value, float) and 1.0 <= value < 1.01


@given(value=from_set(ipv4.cidr("10.0.0.0/8")))
@SETTINGS
def test_draws_from_a_cidr_block(value):
    assert value.packed[0] == 10


@given(value=from_set(ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")))
@SETTINGS
def test_draws_around_a_hole(value):
    assert value.packed[0] == 10 and value.packed[1] != 1
