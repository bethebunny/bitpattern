"""Dates and datetimes, as counts of days and microseconds."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cached_property

from ..intset import IntSet
from .codec import BDDSet, OrderedCodec

__all__ = ["Date", "Datetime", "dates", "datetimes", "naive_datetimes"]

MICROSECOND = dt.timedelta(microseconds=1)
# From datetime.min to datetime.max, in microseconds.
LONGEST = (dt.datetime.max - dt.datetime.min) // MICROSECOND


@dataclass(frozen=True, repr=False)
class Date(OrderedCodec[dt.date]):
    """Dates as their proleptic Gregorian ordinal, where `date.min` is 1.

    Ordinals count up a day at a time, so ranges are contiguous and iteration goes
    forward in time.

    `datetime` subclasses `date`, but a datetime never equals a date, so this refuses
    them. Use `value.date()` to drop the time.
    """

    def encode(self, value: object) -> int:
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return value.toordinal()
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> dt.date:
        return dt.date.fromordinal(bits)

    @cached_property
    def all(self) -> BDDSet[dt.date]:
        """Every date. Ordinal 0 and those past `date.max` aren't dates."""
        stop = self.encode(dt.date.max) + 1
        return BDDSet(self, IntSet.range(1, stop, width=self.width))


@dataclass(frozen=True, repr=False)
class Datetime(OrderedCodec[dt.datetime]):
    """Datetimes as microseconds since `datetime.min`, so they sort in time.

    Aware datetimes count from `datetime.min` in UTC, so sets hold instants, the same
    way Python compares them, and noon in UTC is the same member as 7am in New York.
    They decode into `tz`. With `tz=None` the codec is naive, and counts wall-clock
    time instead. Naive and aware datetimes don't compare, so each kind of codec
    refuses the other.

    In a zone with daylight saving, the hour the clocks skip has no instants, so it
    never decodes, and the hour they repeat decodes twice, the second time with
    `fold=1`. Python counts those two as equal, since they share a zone. A naive codec
    doesn't keep `fold`, since naive datetimes ignore it when they compare.
    """

    tz: dt.tzinfo | None

    @cached_property
    def _epoch(self) -> dt.datetime:
        return dt.datetime.min.replace(tzinfo=None if self.tz is None else dt.UTC)

    def _microseconds(self, value: dt.datetime) -> int:
        return (value - self._epoch) // MICROSECOND

    def encode(self, value: object) -> int:
        # A tzinfo that gives no UTC offset still leaves a datetime naive.
        if (
            isinstance(value, dt.datetime)
            and (value.utcoffset() is None) == (self.tz is None)
            and (microseconds := self._microseconds(value)) in self.all.storage
        ):
            return microseconds
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> dt.datetime:
        moment = self._epoch + bits * MICROSECOND
        return moment if self.tz is None else moment.astimezone(self.tz)

    @cached_property
    def all(self) -> BDDSet[dt.datetime]:
        """Every datetime. An aware codec leaves out instants at the ends that would
        decode into `tz` outside the years datetime can hold."""
        low = max(self._microseconds(dt.datetime.min.replace(tzinfo=self.tz)), 0)
        high = min(self._microseconds(dt.datetime.max.replace(tzinfo=self.tz)), LONGEST)
        return BDDSet(self, IntSet.range(low, high + 1, width=self.width))


# The fewest bits that hold date.max and datetime.max.
dates = Date(22, "dates")
datetimes = Datetime(59, "datetimes", tz=dt.UTC)
naive_datetimes = Datetime(59, "naive_datetimes", tz=None)
