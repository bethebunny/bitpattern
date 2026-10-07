"""Dates and naive datetimes, as counts of days and microseconds."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cached_property

from ..intset import IntSet
from .codec import BDDSet, OrderedCodec

__all__ = ["Date", "Datetime", "dates", "datetimes"]

MICROSECOND = dt.timedelta(microseconds=1)


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

    def pattern(self, text: str) -> BDDSet[dt.date]:
        # Bits can match ordinal 0, or ordinals past date.max, which aren't dates.
        return super().pattern(text) & self.all


@dataclass(frozen=True, repr=False)
class Datetime(OrderedCodec[dt.datetime]):
    """Naive datetimes as microseconds since `datetime.min`.

    Microseconds count up with time, so ranges are contiguous and iteration goes
    forward in time. `fold` isn't kept, since naive datetimes ignore it when they
    compare.

    Aware datetimes are refused, since they don't compare with naive ones.
    `value.astimezone(UTC).replace(tzinfo=None)` converts one to naive UTC.
    """

    def encode(self, value: object) -> int:
        # A tzinfo that gives no UTC offset still leaves a datetime naive.
        if isinstance(value, dt.datetime) and value.utcoffset() is None:
            return (value - dt.datetime.min) // MICROSECOND
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> dt.datetime:
        return dt.datetime.min + bits * MICROSECOND

    @cached_property
    def all(self) -> BDDSet[dt.datetime]:
        """Every naive datetime. Bits past `datetime.max` aren't datetimes."""
        stop = self.encode(dt.datetime.max) + 1
        return BDDSet(self, IntSet.range(0, stop, width=self.width))

    def pattern(self, text: str) -> BDDSet[dt.datetime]:
        # Bits can match microseconds past datetime.max, which aren't datetimes.
        return super().pattern(text) & self.all


# The fewest bits that hold date.max and datetime.max.
dates = Date(22, "dates")
datetimes = Datetime(59, "datetimes")
