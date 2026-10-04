# bitpattern

> [!NOTE]
> ```bash
> pip install bitpattern
> pip install 'bitpattern[hypothesis]'  # with the hypothesis strategies
> ```

bitpattern is a library for describing and working with very large sets of structured
data. It can sample and count sets much larger than would ordinarily fit in memory.

Consider for example 64 bit floats. How many finite normal floats are there? Can we
sample them directly? These sets are huge and noncontiguous, so you can't express them
in traditional data structures.

```python
>>> from bitpattern.codecs import float64
>>> supported = float64.finite - float64.subnormal
>>> supported.size
18428729675200069634
>>> 0.25 in supported
True
>>> list(supported[:3])
[0.0, 2.2250738585072014e-308, 2.225073858507202e-308]
>>> supported.choice()
2.3171912436506223e+167
>>> float64.range(1.0, 2.0).size
4503599627370496

```

`supported` is a `BDDSet[float]`, a set of floats backed by the `IntSet` of their bits,
`supported.storage`. `float64` is the [codec](#codecs) that encodes them. BDDSets work
with the normal set operations, and are also sequences of their values, in the order of
their bits.

bitpattern uses a data structure called a Binary Decision Diagram to encode
extremely large sets. Whereas set operations are typically described in terms
of the size of the set, bitpattern sets support most operations in O(#bits) of
the _largest member_ of the set. #bits is called the `width` of the set.

This becomes particularly useful for use cases like [hypothesis](https://hypothesis.works/).

Hypothesis really likes you to express and sample from the _true domain_ of your
input data. Rejection sampling (in hypothesis literally sampling from the whole
range and then calling `reject` on inputs that don't match your criteria) frequently
eliminates too much data. By default hypothesis will fail tests that reject
more than ~80% of inputs, but for instance subnormals are well under 1% of floats, so
this isn't practical.

`bitpattern.strategies` turns these sets into hypothesis strategies:

```python
from hypothesis import given

from bitpattern.codecs import float64
from bitpattern.strategies import from_set


@given(from_set(float64.finite - float64.subnormal))
def test_kernel_matches_reference(x: float): ...
```

## Patterns

The pattern syntax follows normal glob rules. Patterns are expressed as quartets of 4
bits separated by `.`, most significant to least significant, with 0 and 1 representing
a fixed bit, ? can be either, and * is shorthand for multiple ?. The leading quartet can
be shorter, for widths that aren't a multiple of 4.

Patterns are sets, and work with normal set operations.

```python
>>> from bitpattern import Pattern
>>> Pattern("*1.0000")
Pattern('???1.0000')
>>> Pattern("0000") | Pattern("0001")
Pattern('000?')
>>> ~Pattern("00??")
Pattern('01??') | Pattern('1???')

```

Codecs take patterns too, which match the bits of their encoding. A float64 is a sign
bit, 11 exponent bits and 52 mantissa bits, and quiet NaNs have every exponent bit and
the top mantissa bit set:

```python
>>> quiet = float64.pattern("?111.1111.1111.1*.*.*.*.*.*.*.*.*.*.*.*.*")
>>> quiet <= float64.nan
True

```

## Codecs

A `Codec` encodes values of a type as fixed-width integers, and decodes them back.
Its sets are `BDDSet`s, so patterns match the bits of the encoding. To write one,
subclass `Codec` and implement `encode` and `decode`:

```python
>>> from bitpattern import Codec
>>> class Ascii(Codec[str]):
...     def encode(self, value: object) -> int:
...         if isinstance(value, str) and len(value) == 1 and value.isascii():
...             return ord(value)
...         raise ValueError(f"{value!r} isn't an ASCII character")
...
...     def decode(self, bits: int) -> str:
...         return chr(bits)
>>> ascii = Ascii(7, "ascii")
>>> ascii.set("hello")
BDDSet(ascii, ['e', 'h', 'l', 'o'])
>>> ascii.pattern("1?0.0001")  # case only changes bit 5
BDDSet(ascii, ['A', 'a'])

```

`encode` should raise `ValueError` for anything it can't encode. If some bit patterns
aren't values, override `all` with the ones that are, so that `~` leaves the others out.

`bitpattern.codecs` includes:

- `float16`, `float32` and `float64`, which are IEEE 754 floats as their raw bits, and
  the ML formats `bfloat16`, `float8_e5m2` and `float8_e4m3fn`. They have sets like
  `finite`, `nan`, `infinities`, `subnormal` and `zeros`, and `range(low, high)`.
- `int8`, `int16`, `int32`, `int64` and `int128`, and `uint8` through `uint128`, which
  are integers as their raw bits, in two's complement if they're signed. They have
  `min`, `max` and `range(low, high)`.
- `char`, one-character strings as their code points, for every Unicode scalar
  value, with `range(low, high)` and `category(name)` for general categories like
  `"Lu"` or `"L"`.
- `ipv4` and `ipv6`, with `cidr("10.0.0.0/8")` and `range(low, high)`.
  `networks(addresses)` finds the fewest CIDR blocks that make up a set of addresses.
- `mac`, MAC addresses as strings like `01:23:45:67:89:ab`, with `multicast`,
  `unicast`, `local`, `universal` and `broadcast`, and `oui(prefix)` for the
  addresses under an OUI.
- `uuid128`, `uuid.UUID`s as their 128 bits, with `version(n)` for RFC 9562's
  versions 1 to 8, `rfc9562` for its variant, and `nil` and `max`.
- `dates` and `datetimes`, which count `datetime.date`s in days and naive
  `datetime.datetime`s in microseconds, so they sort in time, with
  `range(low, high)`.
- `Members(SomeEnum)`, an Enum's members by their index, and `Flags(SomeFlag)`, a
  Flag's combinations as their own bits, with `having(flags)` and `lacking(flags)`.

`Tuple` puts codecs side by side, the first field's bits on top. `product` builds a
set of tuples from a set for each field, and costs about as many nodes as they do
together, however many tuples it holds. `project` goes back the other way.

Here's a quick tour, with sampling seeded so it's repeatable:

```python
>>> import random, uuid
>>> from datetime import date, datetime
>>> from enum import Flag
>>> from bitpattern.codecs import (
...     Flags, Tuple, char, dates, datetimes, float8_e4m3fn, ipv4, mac, uint16, uuid128
... )
>>> rng = random.Random(0)

>>> # Tuples: every (address, port) on a private network's unprivileged ports
>>> sockets = Tuple(ipv4, uint16)
>>> private = sockets.product(
...     ipv4.cidr("10.0.0.0/8") | ipv4.cidr("192.168.0.0/16"),
...     uint16.range(1024, 65536),
... )
>>> private.size
1086559617024
>>> ("10.1.2.3", 8080) in private
True
>>> private.choice(rng=rng)
(IPv4Address('10.100.105.211'), 22477)
>>> sockets.project(private, 1) == uint16.range(1024, 65536)
True

>>> # fp8: the whole format, and how floats round into it
>>> float8_e4m3fn.finite.size
254
>>> float8_e4m3fn.positive[-1]
448.0
>>> float8_e4m3fn.decode(float8_e4m3fn.encode(0.3))
0.3125
>>> list(float8_e4m3fn.range(1.0, 2.0))
[1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]

>>> # Unicode characters
>>> "É" in char.category("Lu")
True
>>> "".join(char.category("Sm") & char.range("\0", "\x80"))
'+<=>|~'
>>> char.range("a", "{") == char.category("Ll") & char.range("\0", "\x80")
True

>>> # Random MAC addresses are the locally administered unicast ones
>>> (mac.local & mac.unicast).size
70368744177664
>>> (mac.local & mac.unicast).choice(rng=rng)
'16:2e:6b:aa:94:55'

>>> # UUIDs
>>> uuid128.version(4).size == 2**122
True
>>> uuid.uuid4() in uuid128.version(4)
True
>>> uuid128.version(7).choice(rng=rng)
UUID('f8cb83e0-b8b9-798b-b728-b4fa42485e3a')

>>> # Dates and datetimes
>>> dates.range(date(2024, 1, 1), date(2025, 1, 1)).size
366
>>> datetimes.range(datetime(2024, 1, 1), datetime(2024, 1, 2)).size
86400000000

>>> # Tuples of anything: a year of UUIDv7 events
>>> events = Tuple(uuid128, datetimes)
>>> year = events.product(
...     uuid128.version(7), datetimes.range(datetime(2024, 1, 1), datetime(2025, 1, 1))
... )
>>> year.choice(rng=rng)
(UUID('3e1a790b-610b-7b0c-ad79-9e3519a6aec7'), datetime.datetime(2024, 6, 15, 10, 18, 47, 668228))

>>> # Flags
>>> class Permission(Flag):
...     READ = 4
...     WRITE = 2
...     EXECUTE = 1
>>> permissions = Flags(Permission)
>>> list(permissions.having(Permission.READ))
[<Permission.READ: 4>, <Permission.READ|EXECUTE: 5>, <Permission.READ|WRITE: 6>, <Permission.READ|WRITE|EXECUTE: 7>]
>>> permissions.pattern("?1?") == permissions.having(Permission.WRITE)
True

```

## IntSet

`IntSet` is the backing abstraction for BDDSets and patterns, and is provided directly.
An `IntSet` is an immutable set of
non-negative integers below `2 ** width`. It's a `collections.abc.Set`, and also a
`Sequence` of its members in sorted order. Slicing gives back a set.

An `IntSet` is a reduced, ordered binary decision diagram, or BDD ([Bryant, 1986]).
The BDD abstraction is also provided directly, as `bitpattern.BDD`.

```python
>>> from bitpattern import IntSet
>>> s = IntSet.range(3, 17)
>>> s
IntSet([3, 4, 5, 6, ..., 15, 16], size=14, width=5)
>>> s[2]
5
>>> s[2:5]
IntSet([5, 6, 7], width=5)
>>> s & IntSet([1, 2, 3, 4])
IntSet([3, 4], width=5)
>>> ~IntSet([1, 3], width=2)
IntSet([0, 2], width=2)

```

> [!WARNING]
> Python isn't really designed for data structures larger than memory.
> In particular many operations will fail if `__len__` returns a number >= 2**63.
> When working with very large sets:
>
> - Use `myset.size` instead of `len(myset)`
> - Use `myset.choice()` instead of `random.choice(myset)` or `random.sample(myset, k)`
> - Use `myset[0]` and `myset[-1]` instead of `min(myset)` and `max(myset)`, which
>   look at every member
> - Use `bitpattern.strategies` instead of hypothesis's `sampled_from(myset)`

Here's how `IntSet` compares to other ways of storing a set. `n` and `m` are set
sizes, `w` is the width (number of bits of the largest member), and `|a|` is the number of nodes in `a`'s diagram.
`|a|` is at most `n * w`, and is usually much smaller. Notably, a set written as a single
pattern has `|a| <= w`, since only its fixed bits need nodes. For most use cases `w` is a
constant and may be read as `O(1)`.

| | sorted `list` | `set` | balanced tree | `IntSet` |
| --- | --- | --- | --- | --- |
| `x in a` | O(log n) | O(1) | O(log n) | O(w) |
| `a[i]` | O(1) | n/a | O(log n) | O(w) |
| `len(a)` | O(1) | O(1) | O(1) | O(1) |
| `a \| b`, `a & b`, `a - b` | O(n + m) | O(n + m) | O(n + m) | O(\|a\| · \|b\|) |
| `~a` | O(2<sup>w</sup>) | O(2<sup>w</sup>) | O(2<sup>w</sup>) | O(\|a\|) |
| slice `a[i:j]` | O(j - i) | n/a | O(log n + j - i) | O(w) |
| `a == b` | O(n) | O(n) | O(n) | O(1) |
| `hash(a)` | O(n) | O(n) | O(n) | O(w) |
| random member | O(1) | O(n) | O(log n) | O(w) |
| memory | O(n) | O(n) | O(n) | O(\|a\|) |

`a == b` is O(1) between sets of the same width, and O(w) between different widths.

[Bryant, 1986]: https://doi.org/10.1109/TC.1986.1676819

R. E. Bryant. Graph-Based Algorithms for Boolean Function Manipulation. _IEEE
Transactions on Computers_, 35(8):677–691, 1986.
