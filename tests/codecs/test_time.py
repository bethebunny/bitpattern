"""Date and datetime codecs, checked against the datetime module."""

import itertools
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern.codecs.time import DateTime, dates, datetimes, naive_datetimes

MICROSECOND = timedelta(microseconds=1)
HOUR = 3600 * 10**6  # in microseconds
NEW_YORK = ZoneInfo("America/New_York")
ZONES = [
    datetimes,
    DateTime(NEW_YORK),
    DateTime(timezone(timedelta(hours=14))),
    DateTime(timezone(timedelta(hours=-12))),
]


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
    assert naive_datetimes.encode(value) == microseconds(value)
    assert naive_datetimes.decode(naive_datetimes.encode(value)) == value
    assert value in naive_datetimes.all


@given(a=st.dates(), b=st.dates())
@settings(deadline=None)
def test_later_dates_have_bigger_encodings(a, b):
    assert (a < b) == (dates.encode(a) < dates.encode(b))


@given(a=st.datetimes(), b=st.datetimes())
@settings(deadline=None)
def test_later_datetimes_have_bigger_encodings(a, b):
    assert (a < b) == (naive_datetimes.encode(a) < naive_datetimes.encode(b))


def test_dates_run_from_min_to_max():
    assert dates.encode(date.min) == 1
    assert dates.all[0] == date.min
    assert dates.all[-1] == date.max
    assert dates.all.size == date.max.toordinal()
    assert dates.encode(date.max).bit_length() == dates.width


def test_datetimes_run_from_min_to_max():
    assert naive_datetimes.encode(datetime.min) == 0
    assert naive_datetimes.all[0] == datetime.min
    assert naive_datetimes.all[-1] == datetime.max
    assert naive_datetimes.all.size == (datetime.max - datetime.min) // MICROSECOND + 1
    assert naive_datetimes.encode(datetime.max).bit_length() == naive_datetimes.width


def test_invert_stays_between_min_and_max():
    ends = ~dates.range(date(1, 1, 3), date.max)
    assert list(ends) == [date(1, 1, 1), date(1, 1, 2), date.max]
    assert ~naive_datetimes.range(datetime.min, datetime.max) == naive_datetimes.set(
        [datetime.max]
    )
    assert ~dates.none == dates.all
    assert ~naive_datetimes.none == naive_datetimes.all


def test_patterns_leave_out_bits_that_arent_values():
    assert dates.pattern("??.*.*.*.*.*") == dates.all
    # Ordinal 0 matches too, but isn't a date.
    assert dates.pattern("??.*.*.*.*.0000")[0] == date(1, 1, 16)
    assert naive_datetimes.pattern("???" + ".*" * 14) == naive_datetimes.all


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
        assert list(naive_datetimes.range(low, high)) == expected


@given(low=st.dates(), high=st.dates(), value=st.dates())
@settings(deadline=None)
def test_date_range_holds_the_dates_between(low, high, value):
    span = dates.range(low, high)
    assert (value in span) == (low <= value < high)
    assert span.size == max((high - low).days, 0)


@given(low=st.datetimes(), high=st.datetimes(), value=st.datetimes())
@settings(deadline=None)
def test_datetime_range_holds_the_datetimes_between(low, high, value):
    span = naive_datetimes.range(low, high)
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
        (naive_datetimes, AWARE),
        (naive_datetimes, datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=-8)))),
        (naive_datetimes, date(2020, 1, 1)),
        (naive_datetimes, "2020-01-01T00:00:00"),
        (naive_datetimes, 0),
        (naive_datetimes, None),
    ],
)
def test_time_codecs_refuse_what_they_cant_encode(codec, value):
    with pytest.raises((TypeError, ValueError)):
        codec.encode(value)
    assert value not in codec.all


@pytest.mark.parametrize(
    "codec, low, high",
    [
        (dates, datetime(2020, 1, 1), datetime(2021, 1, 1)),
        (naive_datetimes, date(2020, 1, 1), date(2021, 1, 1)),
        (naive_datetimes, AWARE, AWARE + timedelta(days=1)),
    ],
)
def test_time_ranges_refuse_bounds_they_cant_encode(codec, low, high):
    with pytest.raises((TypeError, ValueError)):
        codec.range(low, high)


def test_a_tzinfo_without_an_offset_leaves_a_datetime_naive():
    value = datetime(2020, 1, 1, tzinfo=NoOffset())
    assert value == datetime(2020, 1, 1)
    assert naive_datetimes.encode(value) == naive_datetimes.encode(datetime(2020, 1, 1))


def test_datetimes_drop_fold():
    """Naive datetimes ignore fold when they compare."""
    later = datetime(2020, 11, 1, 1, 30, fold=1)
    assert later == later.replace(fold=0)
    assert naive_datetimes.encode(later) == naive_datetimes.encode(
        later.replace(fold=0)
    )
    assert naive_datetimes.decode(naive_datetimes.encode(later)).fold == 0


def test_open_ranges_reach_the_ends_of_time():
    assert date.max in dates.range(date(9999, 1, 1))
    assert list(dates.range(stop=date(1, 1, 3))) == [date(1, 1, 1), date(1, 1, 2)]
    assert naive_datetimes.range() == naive_datetimes.all
    assert (dates.min, dates.max) == (date.min, date.max)
    assert (naive_datetimes.min, naive_datetimes.max) == (datetime.min, datetime.max)


@given(
    value=st.datetimes(
        min_value=datetime(2, 1, 1),
        max_value=datetime(9998, 1, 1),
        timezones=st.timezones(),
    )
)
@settings(deadline=None)
def test_aware_datetimes_are_their_instants(value):
    instant = value.astimezone(UTC)
    assert datetimes.encode(value) == microseconds(instant.replace(tzinfo=None))
    assert datetimes.decode(datetimes.encode(value)) == instant
    assert datetimes.decode(datetimes.encode(value)).tzinfo is UTC


def test_an_instant_is_the_same_member_in_any_zone():
    noon = datetime(2024, 1, 1, 12, tzinfo=UTC)
    assert noon.astimezone(ZoneInfo("Asia/Tokyo")) in datetimes.set([noon])
    assert ZONES[1].encode(noon) == datetimes.encode(noon)


def test_naive_and_aware_codecs_refuse_each_others_datetimes():
    with pytest.raises(ValueError):
        datetimes.encode(datetime(2020, 1, 1))
    with pytest.raises(ValueError):
        naive_datetimes.encode(datetime(2020, 1, 1, tzinfo=UTC))


def test_the_hour_clocks_skip_never_decodes():
    """New York's clocks go from 2:00 to 3:00 on 2024-03-10."""
    new_york = ZONES[1]
    start, stop = (datetime(2024, 3, 10, hour, tzinfo=NEW_YORK) for hour in (1, 4))
    morning = new_york.range(start, stop)
    assert morning.size == 2 * HOUR  # three hours of clock, two of time
    assert [value.hour for value in morning[:: HOUR // 2]] == [1, 1, 3, 3]


def test_the_hour_clocks_repeat_decodes_twice():
    """New York's clocks go back from 2:00 to 1:00 on 2024-11-03."""
    new_york = ZONES[1]
    start, stop = (datetime(2024, 11, 3, hour, tzinfo=NEW_YORK) for hour in (0, 2))
    night = list(new_york.range(start, stop)[:: HOUR // 2])
    assert [(value.hour, value.minute, value.fold) for value in night] == [
        (0, 0, 0),
        (0, 30, 0),
        (1, 0, 0),
        (1, 30, 0),
        (1, 0, 1),
        (1, 30, 1),
    ]
    # Python compares datetimes in one timezone by their clocks, so these two are equal.
    assert night[3] == night[5]
    assert night[3].astimezone(UTC) != night[5].astimezone(UTC)


@pytest.mark.parametrize("codec", ZONES, ids=repr)
def test_aware_codecs_decode_every_instant_at_their_ends(codec):
    first, last = codec.all[0], codec.all[-1]
    assert (first.utcoffset(), last.utcoffset()) == (
        codec.tz.utcoffset(first),
        codec.tz.utcoffset(last),
    )
    assert codec.all.size == codec.encode(last) - codec.encode(first) + 1
    assert codec.range() == codec.all


def test_datetime_codecs_are_named_after_their_zone():
    assert [repr(codec) for codec in ZONES] == [
        "datetimes[UTC]",
        "datetimes[America/New_York]",
        "datetimes[UTC+14:00]",
        "datetimes[UTC-12:00]",
    ]
    assert repr(naive_datetimes) == "naive_datetimes"
    assert DateTime(UTC) == datetimes and DateTime(None) == naive_datetimes
    assert DateTime(NEW_YORK).all | ZONES[1].none == ZONES[1].all


def test_only_the_right_kind_of_now_is_a_member():
    """A naive `now()` isn't an instant, just as a naive datetime never equals an
    aware datetime."""
    assert datetime.now() not in datetimes.all
    assert datetime.now(UTC) in datetimes.all
    assert datetime.now() in naive_datetimes.all
