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
    """Dates, in time order.

    `Date` doesn't accept datetimes even though `datetime` subclasses `date`. Use
    `value.date()` to drop the time.
    """

    def encode(self, value: object) -> int:
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return value.toordinal()
        raise ValueError(f"{self} can't encode {value!r}")

    def decode(self, bits: int) -> dt.date:
        return dt.date.fromordinal(bits)

    @cached_property
    def all(self) -> BDDSet[dt.date]:
        """Every date. Ordinal 0 and ordinals past `date.max` aren't dates."""
        stop = self.encode(dt.date.max) + 1
        return BDDSet(self, IntSet.range(1, stop, width=self.width))


@dataclass(frozen=True, repr=False, init=False)
class Datetime(OrderedCodec[dt.datetime]):
    """Datetimes, in time order.

    `Datetime(tz)` holds timezone-aware datetimes and decodes them into `tz`. It
    accepts datetimes from any timezone, eg. 12:00 UTC and 07:00 EST are the same
    member. `Datetime(None)` holds naive datetimes. Python can't compare the two
    kinds, so neither codec accepts the other.

    When `tz` has daylight saving:
      - local times the clocks skip never decode
      - local times the clocks repeat decode twice, the second time with `fold=1`
    Repeated values are equal since they share a timezone.
    """

    tz: dt.tzinfo | None

    def __init__(self, tz: dt.tzinfo | None) -> None:
        name = "naive_datetimes" if tz is None else f"datetimes[{tz}]"
        vars(self).update(width=LONGEST.bit_length(), name=name, tz=tz)

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
        """Every datetime. Instants at either end that overflow `datetime` in `tz`
        are left out."""
        low = max(self._microseconds(dt.datetime.min.replace(tzinfo=self.tz)), 0)
        high = min(self._microseconds(dt.datetime.max.replace(tzinfo=self.tz)), LONGEST)
        return BDDSet(self, IntSet.range(low, high + 1, width=self.width))


dates = Date(22, "dates")  # the fewest bits that hold date.max
datetimes = Datetime(dt.UTC)
naive_datetimes = Datetime(None)
