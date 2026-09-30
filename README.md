# bitpattern

Sets of integers, described by their bits and backed by binary decision diagrams.

A pattern is groups of 4 bits, most significant first. `?` is a free bit, and `*`
frees the rest of its group:

```python
>>> from bitpattern import Pattern
>>> p = Pattern("*1.*.*.0000.1111.?01?")
>>> p
Pattern('???1.????.????.0000.1111.?01?')
>>> p.width, p.size
(24, 8192)
>>> p[0], p[-1]
(1048818, 16773371)

```

That's 8192 integers in 11 nodes, one for each bit that's pinned to 0 or 1. Free
bits don't cost anything. Checking membership, counting and indexing all work on
the diagram, so they never have to enumerate the members.

## Use `size`, not `len`

`len()` has to fit in a `Py_ssize_t`, and these sets often don't. `.size` always
works, and so does everything else on a set, eg. indexing, slicing and `choice()`.
Things that call `len()` themselves, like `random.sample`, only work while it fits.

```python
>>> from bitpattern.codecs import float64
>>> float64.finite.size
18437736874454810624
>>> len(float64.finite)
Traceback (most recent call last):
  ...
OverflowError: cannot fit 'int' into an index-sized integer

```

## Sets

`IntSet` is an immutable, hashable `collections.abc.Set`. It's also a `Sequence`
of its members in order, so it can be indexed and sliced, and slicing gives back a
set:

```python
>>> from bitpattern import IntSet
>>> s = IntSet.range(3, 17)
>>> s
IntSet([3, 4, 5, 6, ..., 15, 16], size=14, width=5)
>>> IntSet([1, 2]) <= IntSet([1, 2, 3])
True
>>> s[2], s.index(10)
(5, 7)
>>> s[2:5]
IntSet([5, 6, 7], width=5)

```

Ranges cost one node per bit, however many integers are in them.
`IntSet.range(10**15, 10**18, width=64)` is 90 nodes.

## Floats

IEEE 754 floats are sign and magnitude, so non-negative floats sort in the same
order as their bits. That makes index 0 the natural minimal example:

```python
>>> reals = float64.positive - float64.infinities
>>> reals.size
9218868437227405311
>>> float64.decode(float64.range(1.0, 2.0)[0])
1.0
>>> positives = float64.finite & float64.sign_clear
>>> float64.decode(positives[0]), float64.decode(positives[1])
(0.0, 5e-324)

```

The named sets are `positive`, `negative`, `nonnegative`, `zeros`, `subnormal`,
`infinities`, `nan` and `finite`, plus `sign_clear` and `sign_set`, which split
every bit pattern in half by its sign bit. NaNs have a sign bit, but they aren't
positive or negative, so a NaN can be in `sign_clear` but never in `positive`.

## IP addresses

```python
>>> from bitpattern.codecs import ipv4
>>> spare = ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")
>>> [str(n) for n in ipv4.networks(spare)][:3]
['10.0.0.0/16', '10.2.0.0/15', '10.4.0.0/14']

```

`networks` reads the fewest CIDR blocks that make up a set straight off the
diagram, so route aggregation comes for free. `ipv6` works the same way, with 128
bits.

## Hypothesis

```bash
pip install 'bitpattern[hypothesis]'
```

```python
from hypothesis import given

from bitpattern.codecs import float64
from bitpattern.strategies import from_codec


@given(from_codec(float64, float64.finite))
def test_round_trips(value: float): ...
```

Values are drawn by index, so drawing from `2**63` floats is just as cheap as
drawing from 3, and shrinking the index shrinks the float.

## Writing sets as patterns

Any set can be written out as a pattern:

```python
>>> len(spare.pattern.branches)
8
>>> spare.pattern.branches[0]
'0000.1010.0000.0000.????.????.????.????'

```

A pattern only spells out a single branch. Unions are Python, and that's how they
print, as the expression that builds them:

```python
>>> Pattern("0000") | Pattern("0001")
Pattern('000?')
>>> ~Pattern("00??")
Pattern('01??') | Pattern('1???')

```

Branches come from the diagram rather than the text, so patterns are canonical:
`Pattern("0000") | Pattern("0001")` _is_ `Pattern("000?")`, and prints that way.

A `Pattern` is an `IntSet` that also knows how to write itself down, so the two
mix freely, and operations on a pattern give back a pattern:

```python
>>> isinstance(Pattern("00??"), IntSet)
True
>>> Pattern("00??") == IntSet([0, 1, 2, 3], 4)
True

```

## Layout

| module | what's in it |
| --- | --- |
| `bitpattern` | `IntSet`, `Pattern` |
| `bitpattern.codecs` | `float16`, `float32`, `float64`, `ipv4`, `ipv6`, `Codec` |
| `bitpattern.strategies` | `from_intset`, `from_codec` |
| `bitpattern.bdd` | `BDD`, the diagrams themselves, if you want to build on them |

MIT licensed. Needs Python 3.11+.
