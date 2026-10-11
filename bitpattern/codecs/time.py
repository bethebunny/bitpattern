"""Dates and datetimes, as counts of days and microseconds."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cached_property

from ..intset import IntSet
from .codec import BDDSet, OrderedCodec

__all__ = ["Date", "DateTime", "dates", "datetimes", "naive_datetimes"]

MICROSECOND = dt.timedelta(microseconds=1)
LARGEST_DATETIME_US = (dt.datetime.max - dt.datetime.min) // MICROSECOND
LARGEST_DATE_DAYS = (dt.date.max - dt.date.min).days


@dataclass(frozen=True, repr=False)
class Date(OrderedCodec[dt.date]):
    """Dates in time order."""

    def encode(self, value: object) -> int:
        if isinstance(value, dt.datetime):
            raise TypeError(
                "Refusing to implicitly encode datetime object. Use `datetime.date()` explicitly."
            )
        if not isinstance(value, dt.date):
            raise TypeError(f"{self} can't encode {value!r}")
        return value.toordinal()

    def decode(self, bits: int) -> dt.date:
        self._check(bits)
        return dt.date.fromordinal(bits)

    @cached_property
    def all(self) -> BDDSet[dt.date]:
        """Every date. Ordinal 0 and ordinals past `date.max` aren't dates."""
        stop = self.encode(dt.date.max) + 1
        return BDDSet(self, IntSet.range(1, stop, width=self.width))


@dataclass(frozen=True, repr=False, init=False)
class DateTime(OrderedCodec[dt.datetime]):
    """datetimes in time order.

    `DateTime(tz)` holds timezone-aware datetimes and decodes them into `tz`.
    - Encoding accepts datetimes from any timezone, eg. 12:00 UTC and 07:00 EST are the same member.
    - `DateTime(None)` holds naive datetimes.
    - naive datetimes and tz-aware datetimes don't accept comparisons, so their respective codecs
        reject encoding the other.

    When `tz` has daylight saving:
      - local times that would never occur have no decoding
      - local times that occur more than once have multiple decodings, distinguished
        with different `fold` values. They compare as equal to each other.
    """

    tz: dt.tzinfo | None

    def __init__(self, tz: dt.tzinfo | None) -> None:
        name = "naive_datetimes" if tz is None else f"datetimes[{tz}]"
        vars(self).update(width=LARGEST_DATETIME_US.bit_length(), name=name, tz=tz)

    @cached_property
    def _epoch(self) -> dt.datetime:
        return dt.datetime.min.replace(tzinfo=None if self.tz is None else dt.UTC)

    def _microseconds(self, value: dt.datetime) -> int:
        return (value - self._epoch) // MICROSECOND

    def encode(self, value: object) -> int:
        if not isinstance(value, dt.datetime):
            raise TypeError(f"{self} can't encode {value!r}")
        # Datetimes having a tzinfo with no UTC offset are naive
        if (value.utcoffset() is None) != (self.tz is None):
            raise ValueError(f"{self} can't encode {value!r}")
        if (microseconds := self._microseconds(value)) not in self.all.storage:
            raise ValueError(f"{self} can't encode {value!r}")
        return microseconds

    def decode(self, bits: int) -> dt.datetime:
        self._check(bits)
        moment = self._epoch + bits * MICROSECOND
        return moment if self.tz is None else moment.astimezone(self.tz)

    @cached_property
    def all(self) -> BDDSet[dt.datetime]:
        """Every datetime. Instants at either end that overflow `datetime` in `tz`
        are omitted."""
        low = max(self._microseconds(dt.datetime.min.replace(tzinfo=self.tz)), 0)
        high = min(
            self._microseconds(dt.datetime.max.replace(tzinfo=self.tz)),
            LARGEST_DATETIME_US,
        )
        return BDDSet(self, IntSet.range(low, high + 1, width=self.width))


dates = Date(LARGEST_DATE_DAYS.bit_length(), "dates")
datetimes = DateTime(dt.UTC)
naive_datetimes = DateTime(None)
