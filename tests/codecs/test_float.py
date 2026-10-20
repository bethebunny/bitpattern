"""Float codecs, with the 8- and 16-bit formats brute-forced over all of their bit
patterns."""

import bisect
import itertools
import math
import pickle
import random
import struct
import sys
from fractions import Fraction
from functools import cache

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern import Pattern
from bitpattern.codecs import Float, bfloat16, float16, float32, float64
from bitpattern.codecs.float import float8_e4m3fn, float8_e5m2

FLOAT8 = [float8_e5m2, float8_e4m3fn]


BRUTE_FORCED = [*FLOAT8, float16, bfloat16]  # small enough to check every bit pattern


SMALL = float16.range(1.0, 1.01)  # the 11 float16s in [1.0, 1.01)


@cache
def decoded(codec: Float) -> list[float]:
    """The value of every bit pattern of a small float codec, indexed by bits."""
    return [codec.decode(bits) for bits in range(1 << codec.width)]


def patterns(codec: Float, predicate) -> set[int]:
    """Every bit pattern of a small float codec whose value satisfies `predicate`."""
    return {bits for bits, value in enumerate(decoded(codec)) if predicate(value)}


def by_definition(codec: Float, bits: int) -> float:
    """`bits` read as a finite float: a sign, an exponent with a bias, and a fraction
    with an implicit leading 1 unless the exponent is 0."""
    sign = -1.0 if bits >> (codec.width - 1) else 1.0
    exponent = (bits >> codec.mantissa) % (1 << codec.exponent)
    fraction = bits % (1 << codec.mantissa) / (1 << codec.mantissa)
    bias = (1 << (codec.exponent - 1)) - 1
    if not exponent:
        return sign * fraction * 2.0 ** (1 - bias)
    return sign * (1 + fraction) * 2.0 ** (exponent - bias)


def signed(value: float) -> tuple[float, float]:
    """`value` with its sign, which orders -0.0 before 0.0 like ranges do."""
    return value, math.copysign(1, value)


@cache
def orders(codec: Float) -> list[tuple[float, float]]:
    """`signed` of every bit pattern of a small float codec."""
    return [signed(value) for value in decoded(codec)]


def brute_force_range(codec: Float, low, high) -> set[int]:
    """Every bit pattern of a small float codec in `[low, high)`, checking each one."""
    start, stop = signed(low), signed(high)
    return {bits for bits, order in enumerate(orders(codec)) if start <= order < stop}


@cache
def magnitudes(codec: Float) -> list[float]:
    """Every finite, non-negative value of a small float codec. Non-negative floats
    count up with their bits, so the list is sorted and indexed by bit pattern."""
    return list(itertools.takewhile(math.isfinite, decoded(codec)))


def bounds(codec: Float) -> list[float]:
    """Values of a small float codec to bound its ranges with, from -inf to inf."""
    finite = magnitudes(codec)
    smallest_normal = finite[1 << codec.mantissa]
    positive = [finite[1], smallest_normal, 1.0, 2.0, finite[-1], math.inf]
    return [-value for value in reversed(positive)] + [-0.0, 0.0] + positive


def between(codec: Float) -> list[float]:
    """Bounds that aren't values of a small float codec: halfway between neighbors,
    where ties round to even, a quarter of the way, and past the largest."""
    finite = magnitudes(codec)
    normal = 1 << codec.mantissa  # where the smallest normal is
    lows = [0, 1, normal - 1, normal, len(finite) - 2]
    halfway = [(finite[at] + finite[at + 1]) / 2 for at in lows]
    quarter = [(3 * finite[at] + finite[at + 1]) / 4 for at in lows]
    positive = [*halfway, *quarter, 2 * finite[-1], sys.float_info.max]
    return [-value for value in reversed(positive)] + positive


def nearest(codec: Float, value: float) -> float | None:
    """`value` rounded to a small float codec by brute force: the closest value, with
    ties going to the even bit pattern. None if `value` rounds up past the largest."""
    finite = magnitudes(codec)
    at = bisect.bisect_left(finite, abs(value))
    if at < len(finite) and finite[at] == abs(value):
        return value
    # Past the largest value, infinity or NaN takes the place of the next one up.
    below = finite[at - 1]
    above = finite[at] if at < len(finite) else 2 * below - finite[-2]
    down = Fraction(abs(value)) - Fraction(below)
    up = Fraction(above) - Fraction(abs(value))
    if down < up or (down == up and (at - 1) % 2 == 0):
        return math.copysign(below, value)
    return math.copysign(above, value) if at < len(finite) else None


def assert_rounds_to_nearest(codec: Float, value: float) -> None:
    """`codec` rounds `value` like `nearest`, and refuses `value` where `nearest`
    gives None."""
    expected = nearest(codec, value)
    if expected is None:
        with pytest.raises(ValueError):
            codec.encode(value)
    else:
        rounded = codec.decode(codec.encode(value))
        assert rounded == expected
        assert math.copysign(1, rounded) == math.copysign(1, expected)


@pytest.mark.parametrize(
    "codec, special",
    [
        (float8_e5m2, Pattern("?111.11??")),
        (float8_e4m3fn, Pattern("?111.1111")),  # only NaNs, with no infinities
        (float16, Pattern("?111.11??.????.????")),
        (bfloat16, Pattern("?111.1111.1???.????")),
    ],
)
def test_decode_agrees_with_the_definition(codec, special):
    """struct decodes the 16-bit formats, so those cases check `by_definition`."""
    for bits, value in enumerate(decoded(codec)):
        if bits in special:
            assert not math.isfinite(value)
        else:
            assert signed(value) == signed(by_definition(codec, bits))


@pytest.mark.parametrize("codec", BRUTE_FORCED)
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
    ],
)
def test_named_sets_match_the_decoded_values(codec, name, predicate):
    assert set(getattr(codec, name).storage) == patterns(codec, predicate)


@pytest.mark.parametrize("codec", BRUTE_FORCED)
def test_subnormals_are_the_values_below_the_smallest_normal(codec):
    smallest_normal = 2.0 ** (2 - 2 ** (codec.exponent - 1))  # 2**(1 - bias)
    below = patterns(codec, lambda v: v != 0 and abs(v) < smallest_normal)
    assert set(codec.subnormal.storage) == below


@pytest.mark.parametrize("codec", BRUTE_FORCED)
def test_sign_halves_partition_everything(codec):
    universe = set(range(1 << codec.width))
    assert set((codec.sign_clear | codec.sign_set).storage) == universe
    assert codec.sign_clear.isdisjoint(codec.sign_set)
    everything = codec.finite | codec.nan | codec.infinities
    assert set(everything.storage) == universe


@pytest.mark.parametrize("codec", BRUTE_FORCED)
def test_nan_isnt_positive_or_negative(codec):
    """NaNs have a sign bit, but no order."""
    assert codec.nan.isdisjoint(codec.positive)
    assert codec.nan.isdisjoint(codec.negative)
    assert not codec.nan.isdisjoint(codec.sign_clear)
    assert not codec.nan.isdisjoint(codec.sign_set)


@pytest.mark.parametrize("codec", [float8_e5m2, float16, bfloat16, float32, float64])
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


def test_float8_e4m3fn_sizes():
    assert float8_e4m3fn.nan.size == 2
    assert float8_e4m3fn.infinities.size == 0
    assert float8_e4m3fn.finite.size == 254
    assert float8_e4m3fn.subnormal.size == 14
    assert float8_e4m3fn.positive.size == 126 == float8_e4m3fn.negative.size


def test_named_sets_are_cached():
    assert float64.nan is float64.nan


def test_key_sorts_bits_by_value():
    ordered = sorted(patterns(float16, lambda v: not math.isnan(v)), key=float16._key)
    values = [float16.decode(bits) for bits in ordered]
    assert all(a <= b for a, b in itertools.pairwise(values))
    assert values[0] == -math.inf and values[-1] == math.inf
    # -0.0 sorts before 0.0, even though -0.0 == 0.0
    assert math.copysign(1, values[len(values) // 2 - 1]) < 0
    assert math.copysign(1, values[len(values) // 2]) > 0


def test_non_negative_bits_count_up_in_value():
    """Counting up in value makes index 0 the natural thing to shrink towards."""
    values = list(float16.finite & float16.sign_clear)
    assert values[0] == 0.0 and math.copysign(1, values[0]) > 0
    assert values[-1] == 65504.0
    assert values == sorted(values)


def test_minus_zero_has_the_highest_bits():
    """`-0.0` is in `nonnegative`, but its sign bit is set."""
    values = list(float16.finite & float16.nonnegative)
    assert math.copysign(1, values[-1]) < 0 and values[-1] == 0.0


@pytest.mark.parametrize("codec", BRUTE_FORCED)
def test_float_range_agrees_with_brute_force(codec):
    for low, high in itertools.product(bounds(codec), repeat=2):
        assert set(codec.range(low, high).storage) == brute_force_range(
            codec, low, high
        )


@pytest.mark.parametrize("codec", BRUTE_FORCED)
def test_float_range_bounds_can_fall_between_floats(codec):
    for bound in between(codec):
        above = codec.range(bound, math.inf)
        assert set(above.storage) == brute_force_range(codec, bound, math.inf)
        below = codec.range(-math.inf, bound)
        assert set(below.storage) == brute_force_range(codec, -math.inf, bound)


def test_float_range_bounds_round_up_to_a_float():
    """400.0 is halfway between 384.0 and 416.0, and ties round to 384.0, the even
    one."""
    assert float8_e4m3fn.range(400.0, math.inf)[0] == 416.0
    assert float8_e4m3fn.range(0.0, 400.0)[-1] == 384.0


def test_float_range_bounds_can_be_past_the_largest_float():
    assert float16.range(0.0, 1e5) == float16.range(0.0, math.inf)
    assert list(float16.range(1e5)) == [math.inf]
    assert float8_e4m3fn.range(1000.0) == float8_e4m3fn.none


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


def test_infinities_bound_float8_e4m3fn_ranges():
    """There aren't any infinities, but a range can still run to one."""
    assert float8_e4m3fn.range(-math.inf, math.inf) == float8_e4m3fn.finite
    assert float8_e4m3fn.range(1.0, math.inf)[-1] == 448.0
    assert float8_e4m3fn.range(math.inf, math.inf) == float8_e4m3fn.none


@pytest.mark.parametrize("codec", [float16, float8_e4m3fn])
def test_nan_cant_bound_a_range(codec):
    with pytest.raises(ValueError):
        codec.range(math.nan, 1.0)
    with pytest.raises(ValueError):
        codec.range(0.0, math.nan)


def test_wide_float_ranges_stay_small():
    """A range is one node per bit, however many floats the range holds."""
    wide = float64.range(-1e300, 1e300)
    assert wide.size > 2**63
    assert len(str(wide.storage.bdd)) < 10_000


def test_float64_range_ends():
    assert float64.range(1.0, 2.0)[0] == 1.0
    unit = float64.range(0.0, 1.0)
    assert unit[0] == 0.0
    assert unit[-1] == math.nextafter(1.0, 0.0)


@pytest.mark.parametrize("codec", [float8_e5m2, float16, bfloat16, float32, float64])
def test_floats_round_trip(codec):
    for value in (0.0, 1.0, -1.0, 2.0, 0.5, math.inf, -math.inf):
        assert codec.decode(codec.encode(value)) == value
    assert math.copysign(1, codec.decode(codec.encode(-0.0))) < 0


@given(value=st.floats(allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_float64_round_trips_any_value(value):
    assert float64.decode(float64.encode(value)) == value


@pytest.mark.parametrize("codec", BRUTE_FORCED)
def test_every_bit_pattern_round_trips(codec):
    for bits, value in enumerate(decoded(codec)):
        assert math.isnan(value) or codec.encode(value) == bits


def test_float8_e4m3fn_nans_round_trip_too():
    """There's one NaN of each sign, so every bit pattern round trips."""
    encoded = [float8_e4m3fn.encode(value) for value in decoded(float8_e4m3fn)]
    assert encoded == list(range(256))


@pytest.mark.parametrize("codec", BRUTE_FORCED)
@given(value=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=300, deadline=None)
def test_encode_rounds_to_the_nearest_value(codec, value):
    assert_rounds_to_nearest(codec, value)


@pytest.mark.parametrize("codec", FLOAT8)
def test_float8_rounding_agrees_with_brute_force(codec):
    """Every value and every midpoint, with a step either side of each midpoint."""
    finite = magnitudes(codec)
    past = 2 * finite[-1] - finite[-2]  # the next step after the largest value
    for below, above in itertools.pairwise([*finite, past]):
        midpoint = (below + above) / 2
        for value in (
            below,
            math.nextafter(midpoint, 0.0),
            midpoint,
            math.nextafter(midpoint, math.inf),
        ):
            assert_rounds_to_nearest(codec, value)
            assert_rounds_to_nearest(codec, -value)


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


def test_float8_e5m2_values():
    assert float8_e5m2.decode(0b0_11110_11) == 57344.0  # the largest
    assert float8_e5m2.positive[0] == 2.0**-16  # the smallest subnormal
    assert float8_e5m2.decode(0b0_00001_00) == 2.0**-14  # the smallest normal
    assert float8_e5m2.encode(1.0) == 0b0_01111_00
    assert float8_e5m2.encode(-math.inf) == 0b1_11111_00
    assert float8_e5m2.encode(61439.0) == 0b0_11110_11  # just short of halfway to inf


def test_float8_e4m3fn_values():
    assert float8_e4m3fn.positive[-1] == 448.0  # the largest
    assert float8_e4m3fn.positive[0] == 2.0**-9  # the smallest subnormal
    assert float8_e4m3fn.decode(0b0_0001_000) == 2.0**-6  # the smallest normal
    assert float8_e4m3fn.decode(0b0_1111_000) == 256.0  # the top exponent is normal
    assert float8_e4m3fn.encode(1.0) == 0b0_0111_000
    assert float8_e4m3fn.encode(-math.nan) == 0b1_1111_111
    # A tie between 448 and the NaN goes to 448, whose mantissa is even.
    assert float8_e4m3fn.encode(464.0) == 0b0_1111_110


def test_encode_refuses_floats_too_big_for_the_format():
    with pytest.raises(ValueError):
        float16.encode(1e10)


@pytest.mark.parametrize(
    "codec, value", [(float16, "x"), (float16, None), (float8_e4m3fn, "x")]
)
def test_floats_refuse_what_isnt_a_number(codec, value):
    with pytest.raises(TypeError):
        codec.encode(value)
    assert value not in codec.all


@pytest.mark.parametrize(
    "codec, value",
    [
        (float8_e5m2, 61440.0),  # rounds to the next step up, which is infinity
        (float8_e4m3fn, math.inf),
        (float8_e4m3fn, -math.inf),
        (float8_e4m3fn, math.nextafter(464.0, math.inf)),  # just past the NaN tie
        (float8_e4m3fn, sys.float_info.max),  # rounding the max up overflows a double
    ],
)
def test_float8_refuses_what_it_cant_encode(codec, value):
    with pytest.raises(ValueError):
        codec.encode(value)
    assert value not in codec.all


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


def test_float_ranges_can_leave_an_end_open():
    assert float16.range() == float16.all - float16.nan
    assert math.inf in float16.range(1.0)
    assert -math.inf in float16.range(stop=0.0)
    assert float8_e4m3fn.range() == float8_e4m3fn.finite
    assert 448.0 in float8_e4m3fn.range(1.0)


@pytest.mark.parametrize(
    "codec, largest",
    [
        (float16, 65504.0),
        (bfloat16, 3.3895313892515355e38),
        (float64, sys.float_info.max),
        (float8_e5m2, 57344.0),
        (float8_e4m3fn, 448.0),
    ],
)
def test_float_min_and_max_are_the_finite_ends(codec, largest):
    assert (codec.min, codec.max) == (-largest, largest)


def rounded_then_packed(codec, value):
    """encode as it was for every format: round to the format, then pack."""
    try:
        packed = struct.pack(codec.format, codec._round(value))
    except OverflowError:
        return None
    return int.from_bytes(packed) >> codec._spare


def encoded(codec, value):
    try:
        return codec.encode(value)
    except ValueError:
        return None


def test_float16_rounds_once_like_it_rounded_twice():
    """struct rounds a double to float16 exactly once, so encode can skip _round:
    the two agree at every float16, halfway between neighbours, and either side."""
    values = [float16.decode(bits) for bits in range(1 << 16)]
    finite = sorted({value for value in values if math.isfinite(value)})
    halves = [(low + high) / 2 for low, high in itertools.pairwise(finite)]
    beside = [
        math.nextafter(half, side) for half in halves for side in (-math.inf, math.inf)
    ]
    for value in [*values, *halves, *beside, 65520.0, -65520.0, 1e300]:
        assert encoded(float16, value) == rounded_then_packed(float16, value), value


@settings(max_examples=2000)
@given(st.floats())
def test_wider_floats_round_once_like_they_rounded_twice(value):
    for codec in (float32, float64):
        assert encoded(codec, value) == rounded_then_packed(codec, value)


def test_float_sets_pickle_after_their_codec_has_encoded():
    float64.encode(0.5)
    group = float64.range(0.0, 1.0)
    assert pickle.loads(pickle.dumps(group)) == group


class Fussy(float):
    def __bool__(self) -> bool:
        raise AssertionError("encode ran the subclass's own code")


@pytest.mark.parametrize("codec", [float64, bfloat16, float8_e4m3fn])
def test_float_subclasses_encode_as_their_doubles(codec):
    assert codec.encode(Fussy(0.5)) == codec.encode(0.5)


@pytest.mark.parametrize("codec", [float64, float16, float8_e4m3fn])
def test_ints_too_big_for_a_double_are_value_errors(codec):
    with pytest.raises(ValueError):
        codec.encode(10**400)
    assert codec.encode(True) == codec.encode(1.0)
