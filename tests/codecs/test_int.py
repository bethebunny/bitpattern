"""Integer codecs, with the 8-bit ones brute-forced."""

import pytest

from bitpattern.codecs import int8, int64, int128, uint8, uint128

INT_BOUNDS = (-1000, -129, -128, -127, -1, 0, 1, 126, 127, 128, 255, 256, 1000)


class Index:
    """Not an int, but has a lossless conversion to one, like numpy's integers."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


def test_int_bounds():
    assert (int8.min, int8.max) == (-128, 127)
    assert (uint8.min, uint8.max) == (0, 255)
    assert (int128.min, int128.max) == (-(2**127), 2**127 - 1)
    assert (uint128.min, uint128.max) == (0, 2**128 - 1)


@pytest.mark.parametrize("codec", [int8, uint8])
def test_ints_use_every_bit_pattern_once(codec):
    values = range(codec.min, codec.max + 1)
    assert sorted(map(codec.encode, values)) == list(range(1 << codec.width))
    assert [codec.decode(codec.encode(value)) for value in values] == list(values)


def test_signed_ints_are_twos_complement():
    assert [int8.encode(value) for value in (0, 1, 127, -128, -1)] == [
        0,
        1,
        127,
        128,
        255,
    ]
    assert int8.pattern("1*.*") == int8.range(int8.min, 0)  # the sign bit


def test_ints_count_up_from_zero_then_from_min():
    assert list(int8.range(-3, 3)) == [0, 1, 2, -3, -2, -1]
    assert list(uint8.range(253, 300)) == [253, 254, 255]


@pytest.mark.parametrize("codec", [int8, uint8])
@pytest.mark.parametrize("low", INT_BOUNDS)
def test_int_range_agrees_with_brute_force(codec, low):
    values = range(codec.min, codec.max + 1)
    for high in INT_BOUNDS:
        assert set(codec.range(low, high)) == {v for v in values if low <= v < high}


def test_wide_int_ranges():
    assert uint128.range(0, 2**100).size == 2**100
    assert int64.range(-(2**40), 2**40).size == 2**41
    assert int128.range(int128.min, int128.max + 1) == int128.all


@pytest.mark.parametrize(
    "codec, value",
    [(int8, 128), (int8, -129), (uint8, -1), (uint8, 256), (int8, 1.0), (int8, "1")],
)
def test_ints_refuse_what_they_cant_encode(codec, value):
    with pytest.raises(ValueError):
        codec.encode(value)
    assert value not in codec.all


def test_ints_take_anything_with_an_index():
    assert int8.encode(Index(-1)) == 255
    assert Index(5) in uint8.range(0, 10)


def test_int_ranges_can_leave_an_end_open():
    assert int8.range() == int8.all
    assert int8.range(stop=0) == int8.pattern("1*.*")
    assert int8.range(100) == int8.range(100, 128)
    assert list(uint8.range(254)) == [254, 255]
