"""Driving hypothesis from an `IntSet`.

This is the use case the library was built for, kept here rather than shipped so
the package stays dependency-free. `from_intset` is the whole integration: an
`IntSet` can report its cardinality and return its n-th member in O(width), so a
strategy is just an index drawn over that range. Nothing is enumerated, so a set
of `2 ** 63` floats costs no more to sample from than a set of three.

The strategies themselves live in `bitpattern.strategies`, behind the optional
`[hypothesis]` extra; this module is the end-to-end proof that they work.
"""

import math

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from bitpattern import IntSet, Pattern
from bitpattern.codecs import float64, ipv4


from bitpattern.strategies import from_codec, from_intset


SETTINGS = settings(max_examples=200, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


class TestFromIntSet:
    @given(value=from_intset(IntSet([3, 99, 12345], 16)))
    @SETTINGS
    def test_draws_only_members(self, value):
        assert value in {3, 99, 12345}

    @given(value=from_intset(IntSet.range(10, 20)))
    @SETTINGS
    def test_respects_a_range(self, value):
        assert 10 <= value < 20

    @given(value=from_intset(Pattern("*1.*.*.0000.1111.?01?").set))
    @SETTINGS
    def test_respects_a_pattern(self, value):
        assert value in Pattern("*1.*.*.0000.1111.?01?")

    @given(value=from_intset(IntSet.range(0, 1 << 64, width=64)))
    @SETTINGS
    def test_draws_from_a_universe_larger_than_a_ssize_t(self, value):
        """The set has 2**64 members, so `len` would not even be expressible."""
        assert 0 <= value < 2**64

    def test_an_empty_set_draws_nothing(self):
        assert from_intset(IntSet()).is_empty


class TestFloats:
    """The motivating example: `float64.positive - nan - inf`, sampled."""

    reals = float64.positive - float64.infinities

    @given(value=from_codec(float64, reals))
    @SETTINGS
    def test_draws_finite_positive_floats(self, value):
        assert math.isfinite(value) and value > 0

    @given(value=from_codec(float64, float64.range(1.0, 2.0)))
    @SETTINGS
    def test_respects_a_value_range(self, value):
        assert 1.0 <= value < 2.0

    @given(value=from_codec(float64, float64.subnormal))
    @SETTINGS
    def test_draws_subnormals(self, value):
        assert value != 0 and abs(value) < 2.2250738585072014e-308

    @given(value=from_codec(float64, float64.nan))
    @SETTINGS
    def test_draws_nans(self, value):
        assert value != value

    def test_sampling_does_not_go_through_len(self):
        """One sign's worth of floats still fits a `Py_ssize_t`; both do not."""
        assert self.reals.size == 2**63 - 2**52 - 1 == len(self.reals)
        assert float64.finite.size == 2**64 - 2**53
        with pytest.raises(OverflowError):
            len(float64.finite)

    @given(value=from_codec(float64, float64.finite))
    @SETTINGS
    def test_draws_from_a_set_len_cannot_measure(self, value):
        assert math.isfinite(value)

    def test_index_zero_is_the_natural_shrink_target(self):
        """Shrinking the index shrinks the float, because the orders agree."""
        positives = float64.finite & float64.sign_clear
        assert float64.decode(positives[0]) == 0.0
        assert float64.decode(positives[1]) == 5e-324


class TestAddresses:
    @given(value=from_codec(ipv4, ipv4.cidr("10.0.0.0/8")))
    @SETTINGS
    def test_draws_from_a_cidr_block(self, value):
        assert value.packed[0] == 10

    @given(value=from_codec(ipv4, ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")))
    @SETTINGS
    def test_respects_a_hole(self, value):
        assert value.packed[0] == 10 and value.packed[1] != 1


class TestStrategyMechanics:
    @given(data=st.data())
    @SETTINGS
    def test_every_member_is_reachable(self, data):
        source = IntSet([1, 2, 3], 4)
        assert data.draw(from_intset(source)) in source

    @given(members=st.sets(st.integers(0, 255), min_size=1, max_size=20), data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_draws_from_any_set(self, members, data):
        assert data.draw(from_intset(IntSet(members))) in members


class TestOptionalExtra:
    """The strategies module is importable without hypothesis installed."""

    def test_importing_the_module_does_not_need_hypothesis(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "hypothesis", None)
        monkeypatch.setitem(sys.modules, "hypothesis.strategies", None)
        monkeypatch.delitem(sys.modules, "bitpattern.strategies", raising=False)
        import importlib

        importlib.import_module("bitpattern.strategies")  # must not raise

    def test_using_it_without_the_extra_says_so(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "hypothesis", None)
        monkeypatch.setitem(sys.modules, "hypothesis.strategies", None)
        with pytest.raises(ImportError, match=r"bitpattern\[hypothesis\]"):
            from_intset(IntSet([1, 2, 3]))
