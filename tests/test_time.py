"""Date and datetime codecs, checked against the datetime module."""

import itertools
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern.codecs.time import dates, datetimes

MICROSECOND = timedelta(microseconds=1)


AWARE = datetime(2020, 1, 1, tzinfo=UTC)


# Where each window of 31 values starts. The middle ones cross a leap day and midnight.
DATE_WINDOWS = (date.min, date(2020, 2, 15), date.max - timedelta(days=30))
DATETIME_WINDOWS = (
    datetime.min,
    datetime(1999, 12, 31, 23, 59, 59, 999_985),
    datetime.max - 30 * MICROSECOND,
)


class NoOffset(tzinfo):
    """A tzinfo that doesn't give a UTC offset, which leaves its datetimes naive."""

    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None

    def tzname(self, dt):
        return None


def microseconds(value: datetime) -> int:
    """`value` in microseconds since `datetime.min`, worked out from its fields."""
    days = value.toordinal() - 1
    seconds = ((days * 24 + value.hour) * 60 + value.minute) * 60 + value.second
    return seconds * 1_000_000 + value.microsecond


@given(value=st.dates())
@settings(deadline=None)
def test_dates_are_their_ordinals(value):
    assert dates.encode(value) == value.toordinal()
    assert dates.decode(dates.encode(value)) == value
    assert value in dates.all


@given(value=st.datetimes())
@settings(deadline=None)
def test_datetimes_are_microseconds_since_min(value):
    assert datetimes.encode(value) == microseconds(value)
    assert datetimes.decode(datetimes.encode(value)) == value
    assert value in datetimes.all


@given(a=st.dates(), b=st.dates())
@settings(deadline=None)
def test_later_dates_have_bigger_encodings(a, b):
    assert (a < b) == (dates.encode(a) < dates.encode(b))


@given(a=st.datetimes(), b=st.datetimes())
@settings(deadline=None)
def test_later_datetimes_have_bigger_encodings(a, b):
    assert (a < b) == (datetimes.encode(a) < datetimes.encode(b))


def test_dates_run_from_min_to_max():
    assert dates.encode(date.min) == 1
    assert dates.all[0] == date.min
    assert dates.all[-1] == date.max
    assert dates.all.size == date.max.toordinal()
    assert dates.encode(date.max).bit_length() == dates.width


def test_datetimes_run_from_min_to_max():
    assert datetimes.encode(datetime.min) == 0
    assert datetimes.all[0] == datetime.min
    assert datetimes.all[-1] == datetime.max
    assert datetimes.all.size == (datetime.max - datetime.min) // MICROSECOND + 1
    assert datetimes.encode(datetime.max).bit_length() == datetimes.width


def test_invert_stays_between_min_and_max():
    ends = ~dates.range(date(1, 1, 3), date.max)
    assert list(ends) == [date(1, 1, 1), date(1, 1, 2), date.max]
    assert ~datetimes.range(datetime.min, datetime.max) == datetimes.set([datetime.max])
    assert ~dates.none == dates.all
    assert ~datetimes.none == datetimes.all


def test_patterns_leave_out_bits_that_arent_values():
    assert dates.pattern("??.*.*.*.*.*") == dates.all
    # Ordinal 0 matches too, but isn't a date.
    assert dates.pattern("??.*.*.*.*.0000")[0] == date(1, 1, 16)
    assert datetimes.pattern("???" + ".*" * 14) == datetimes.all


@pytest.mark.parametrize("start", DATE_WINDOWS)
def test_date_range_agrees_with_brute_force(start):
    window = [start + timedelta(days=days) for days in range(31)]
    for low, high in itertools.product(window[::3], repeat=2):
        assert list(dates.range(low, high)) == [d for d in window if low <= d < high]


@pytest.mark.parametrize("start", DATETIME_WINDOWS)
def test_datetime_range_agrees_with_brute_force(start):
    window = [start + n * MICROSECOND for n in range(31)]
    for low, high in itertools.product(window[::3], repeat=2):
        expected = [d for d in window if low <= d < high]
        assert list(datetimes.range(low, high)) == expected


@given(low=st.dates(), high=st.dates(), value=st.dates())
@settings(deadline=None)
def test_date_range_holds_the_dates_between(low, high, value):
    span = dates.range(low, high)
    assert (value in span) == (low <= value < high)
    assert span.size == max((high - low).days, 0)


@given(low=st.datetimes(), high=st.datetimes(), value=st.datetimes())
@settings(deadline=None)
def test_datetime_range_holds_the_datetimes_between(low, high, value):
    span = datetimes.range(low, high)
    assert (value in span) == (low <= value < high)
    assert span.size == max((high - low) // MICROSECOND, 0)


def test_a_datetime_isnt_in_a_set_of_dates():
    """datetime subclasses date, but a datetime never equals a date."""
    midnight = datetime(2020, 1, 1)
    assert midnight not in {date(2020, 1, 1)}
    assert midnight not in dates.set([date(2020, 1, 1)])


@pytest.mark.parametrize(
    "codec, value",
    [
        (dates, datetime(2020, 1, 1)),
        (dates, "2020-01-01"),
        (dates, date(2020, 1, 1).toordinal()),
        (dates, None),
        (datetimes, AWARE),
        (datetimes, datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=-8)))),
        (datetimes, date(2020, 1, 1)),
        (datetimes, "2020-01-01T00:00:00"),
        (datetimes, 0),
        (datetimes, None),
    ],
)
def test_time_codecs_refuse_what_they_cant_encode(codec, value):
    with pytest.raises(ValueError):
        codec.encode(value)
    assert value not in codec.all


@pytest.mark.parametrize(
    "codec, low, high",
    [
        (dates, datetime(2020, 1, 1), datetime(2021, 1, 1)),
        (datetimes, date(2020, 1, 1), date(2021, 1, 1)),
        (datetimes, AWARE, AWARE + timedelta(days=1)),
    ],
)
def test_time_ranges_refuse_bounds_they_cant_encode(codec, low, high):
    with pytest.raises(ValueError):
        codec.range(low, high)


def test_a_tzinfo_without_an_offset_leaves_a_datetime_naive():
    value = datetime(2020, 1, 1, tzinfo=NoOffset())
    assert value == datetime(2020, 1, 1)
    assert datetimes.encode(value) == datetimes.encode(datetime(2020, 1, 1))


def test_datetimes_drop_fold():
    """Naive datetimes ignore fold when they compare."""
    later = datetime(2020, 11, 1, 1, 30, fold=1)
    assert later == later.replace(fold=0)
    assert datetimes.encode(later) == datetimes.encode(later.replace(fold=0))
    assert datetimes.decode(datetimes.encode(later)).fold == 0
