"""Float codecs, with the 16-bit formats brute-forced over all 65536 of their bit
patterns."""

import bisect
import itertools
import math
import random
from fractions import Fraction
from functools import cache

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern.codecs import Float, bfloat16, float16, float32, float64

UNIVERSE = range(1 << float16.width)


DECODED = [float16.decode(bits) for bits in UNIVERSE]


SMALLEST_NORMAL = 2.0**-14  # float16 has a 5-bit exponent, with a bias of 15


SMALL = float16.range(1.0, 1.01)  # the 10 float16s from 1.0


BOUNDS = (
    -math.inf,
    -65504.0,
    -1.0,
    -SMALLEST_NORMAL,
    -5.960464477539063e-08,
    -0.0,
    0.0,
    5.960464477539063e-08,
    SMALLEST_NORMAL,
    1.0,
    2.0,
    65504.0,
    math.inf,
)


def patterns(predicate) -> set[int]:
    """Every float16 bit pattern whose value satisfies `predicate`."""
    return {bits for bits, value in zip(UNIVERSE, DECODED) if predicate(value)}


@cache
def keys(codec: Float) -> list[int]:
    """`codec._key` of every bit pattern of a 16-bit float codec."""
    return [codec._key(bits) for bits in UNIVERSE]


def brute_force_range(codec: Float, low, high) -> set[int]:
    """Every bit pattern of a 16-bit float codec in `[low, high)`, checking each one."""
    start, stop = codec._key(codec.encode(low)), codec._key(codec.encode(high))
    return {bits for bits, key in zip(UNIVERSE, keys(codec)) if start <= key < stop}


@cache
def magnitudes(codec: Float) -> list[float]:
    """Every finite, non-negative value of a 16-bit float codec. They count up in
    value with their bits, so the list is sorted, and indexed by bit pattern."""
    return [codec.decode(bits) for bits in range(codec.encode(math.inf))]


def nearest(codec: Float, value: float) -> float | None:
    """`value` rounded to a 16-bit float codec by brute force: the closest value, with
    ties going to the even bit pattern. None if it rounds up past the largest."""
    finite = magnitudes(codec)
    at = bisect.bisect_left(finite, abs(value))
    if at < len(finite) and finite[at] == abs(value):
        return value
    # Past the largest value, infinity takes the place of the next one up.
    below = finite[at - 1]
    above = finite[at] if at < len(finite) else 2 * below - finite[-2]
    down = Fraction(abs(value)) - Fraction(below)
    up = Fraction(above) - Fraction(abs(value))
    if down < up or (down == up and (at - 1) % 2 == 0):
        return math.copysign(below, value)
    return math.copysign(above, value) if at < len(finite) else None


@pytest.mark.parametrize(
    "name, predicate",
    [
        ("nan", math.isnan),
        ("infinities", math.isinf),
        ("finite", math.isfinite),
        ("zeros", lambda v: v == 0),
        ("positive", lambda v: v > 0),
        ("negative", lambda v: v < 0),
        ("nonnegative", lambda v: v >= 0),
        ("sign_clear", lambda v: math.copysign(1, v) > 0),
        ("sign_set", lambda v: math.copysign(1, v) < 0),
        ("subnormal", lambda v: v != 0 and abs(v) < SMALLEST_NORMAL),
    ],
)
def test_named_sets_match_the_decoded_values(name, predicate):
    assert set(getattr(float16, name).storage) == patterns(predicate)


def test_sign_halves_partition_everything():
    assert set((float16.sign_clear | float16.sign_set).storage) == set(UNIVERSE)
    assert float16.sign_clear.isdisjoint(float16.sign_set)
    everything = float16.finite | float16.nan | float16.infinities
    assert set(everything.storage) == set(UNIVERSE)


def test_nan_isnt_positive_or_negative():
    """NaNs have a sign bit, but no order."""
    assert float16.nan.isdisjoint(float16.positive)
    assert float16.nan.isdisjoint(float16.negative)
    assert not float16.nan.isdisjoint(float16.sign_clear)
    assert not float16.nan.isdisjoint(float16.sign_set)


@pytest.mark.parametrize("codec", [float16, bfloat16, float32, float64])
def test_named_set_sizes(codec):
    half, mantissa = 1 << (codec.width - 1), 1 << codec.mantissa
    assert codec.all.size == 1 << codec.width
    assert codec.sign_clear.size == half == codec.sign_set.size
    assert codec.zeros.size == 2
    assert codec.infinities.size == 2
    assert codec.nan.size == 2 * mantissa - 2
    assert codec.subnormal.size == 2 * mantissa - 2
    assert codec.positive.size == half - mantissa == codec.negative.size
    assert codec.nonnegative.size == half - mantissa + 2
    assert codec.finite.size == (1 << codec.width) - 2 * mantissa


def test_float64_sizes():
    assert float64.nan.size == 2**53 - 2
    assert (float64.all - float64.nan - float64.infinities).size == 2**64 - 2**53
    assert float64.positive.size == 2**63 - 2**52


def test_named_sets_are_cached():
    assert float64.nan is float64.nan


def test_key_sorts_bits_by_value():
    ordered = sorted(patterns(lambda v: not math.isnan(v)), key=float16._key)
    values = [float16.decode(bits) for bits in ordered]
    assert all(a <= b for a, b in itertools.pairwise(values))
    assert values[0] == -math.inf and values[-1] == math.inf
    # -0.0 sorts before 0.0, even though they're ==
    assert math.copysign(1, values[len(values) // 2 - 1]) < 0
    assert math.copysign(1, values[len(values) // 2]) > 0


def test_non_negative_bits_count_up_in_value():
    """That's what makes index 0 the natural thing to shrink towards."""
    values = list(float16.finite & float16.sign_clear)
    assert values[0] == 0.0 and math.copysign(1, values[0]) > 0
    assert values[-1] == 65504.0
    assert values == sorted(values)


def test_minus_zero_has_the_highest_bits():
    """`-0.0` is in `nonnegative`, but its sign bit is set."""
    values = list(float16.finite & float16.nonnegative)
    assert math.copysign(1, values[-1]) < 0 and values[-1] == 0.0


@pytest.mark.parametrize("codec", [float16, bfloat16])
@pytest.mark.parametrize("low", BOUNDS)
def test_float_range_agrees_with_brute_force(codec, low):
    for high in BOUNDS:
        assert set(codec.range(low, high).storage) == brute_force_range(
            codec, low, high
        )


def test_float_range_is_half_open():
    assert 1.0 in float16.range(1.0, 2.0)
    assert 2.0 not in float16.range(1.0, 2.0)
    assert float16.range(1.0, 1.0).size == 0
    assert float16.range(2.0, 1.0).size == 0


def test_float_range_never_includes_nan():
    spread = float16.range(-math.inf, math.inf)
    assert spread.isdisjoint(float16.nan)
    assert -math.inf in spread
    assert math.inf not in spread


def test_nan_cant_bound_a_range():
    with pytest.raises(ValueError):
        float16.range(math.nan, 1.0)
    with pytest.raises(ValueError):
        float16.range(0.0, math.nan)


def test_wide_float_ranges_stay_small():
    """A range is one node per bit, however many floats are in it."""
    wide = float64.range(-1e300, 1e300)
    assert wide.size > 2**63
    assert len(str(wide.storage.bdd)) < 10_000


def test_float64_range_ends():
    assert float64.range(1.0, 2.0)[0] == 1.0
    unit = float64.range(0.0, 1.0)
    assert unit[0] == 0.0
    assert unit[-1] == math.nextafter(1.0, 0.0)


@pytest.mark.parametrize("codec", [float16, bfloat16, float32, float64])
def test_floats_round_trip(codec):
    for value in (0.0, 1.0, -1.0, 2.0, 0.5, math.inf, -math.inf):
        assert codec.decode(codec.encode(value)) == value
    assert math.copysign(1, codec.decode(codec.encode(-0.0))) < 0


@given(value=st.floats(allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_float64_round_trips_any_value(value):
    assert float64.decode(float64.encode(value)) == value


def test_bfloat16_round_trips_every_bit_pattern():
    for bits in UNIVERSE:
        value = bfloat16.decode(bits)
        assert math.isnan(value) or bfloat16.encode(value) == bits


@pytest.mark.parametrize("codec", [float16, bfloat16])
@given(value=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=300, deadline=None)
def test_encode_rounds_to_the_nearest_value(codec, value):
    expected = nearest(codec, value)
    if expected is None:
        with pytest.raises(ValueError):
            codec.encode(value)
    else:
        rounded = codec.decode(codec.encode(value))
        assert rounded == expected
        assert math.copysign(1, rounded) == math.copysign(1, expected)


@pytest.mark.parametrize("codec", [float16, bfloat16])
@given(data=st.data())
@settings(max_examples=300, deadline=None)
def test_encode_rounds_values_by_a_midpoint(codec, data):
    """Values a step either side of a tie are where rounding twice goes wrong."""
    finite = magnitudes(codec)
    at = data.draw(st.integers(1, len(finite) - 1))
    midpoint = (finite[at - 1] + finite[at]) / 2
    side = data.draw(st.sampled_from([0.0, midpoint, math.inf]))
    value = data.draw(st.sampled_from([1, -1])) * math.nextafter(midpoint, side)
    assert codec.decode(codec.encode(value)) == nearest(codec, value)


def test_bfloat16_rounds_once():
    """Through a float32, this would land exactly on the midpoint, then tie down to 1."""
    assert bfloat16.decode(bfloat16.encode(1 + 2**-8 + 2**-30)) == 1 + 2**-7
    assert bfloat16.decode(bfloat16.encode(1 + 2**-8)) == 1  # a real tie goes to even


@pytest.mark.parametrize("value", ["x", None, 1e10])  # 1e10 is too big for a float16
def test_encode_refuses_what_it_cant_encode(value):
    with pytest.raises(ValueError):
        float16.encode(value)


def test_float_choice_is_always_a_member():
    source = float16.finite & float16.positive
    rng = random.Random(0)
    for _ in range(200):
        value = source.choice(rng=rng)
        assert math.isfinite(value) and value > 0
        assert value in source


def test_float_choice_reaches_the_whole_set():
    seen = {SMALL.choice(rng=random.Random(seed)) for seed in range(100)}
    assert seen == set(SMALL)


def test_a_custom_float_format():
    assert Float(width=16, name="float16", exponent=5, format=">e") == float16
