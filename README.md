# bitpattern

Sets of integers described by their bits, backed by binary decision diagrams.

A pattern is quartets of bits, most significant first, where `?` frees one bit and
`*` frees the rest of its quartet:

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

That set has 8192 members and is stored in **11 nodes** — one per *pinned* bit, and
nothing at all per free bit. Membership, cardinality and n-th-element are all O(width),
so nothing here ever enumerates.

## `size`, not `len`

`len()` must fit a `Py_ssize_t`, and these sets routinely do not. Use `.size` whenever
the universe is wider than 63 bits; everything else — indexing, iteration, `sample` —
works regardless.

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

`IntSet` is an immutable, hashable `collections.abc.Set`:

```python
>>> from bitpattern import IntSet
>>> IntSet.range(3, 17)
IntSet([3, 4, 5, 6, ..., 15, 16], size=14, width=5)
>>> IntSet([1, 2]) <= IntSet([1, 2, 3])
True

```

Ranges cost one node per bit, not one per member: `IntSet.range(10**15, 10**18, width=64)`
is 90 nodes.

## Floats

IEEE-754 is sign-magnitude, so for non-negative floats the bit order *is* the value
order — which makes index 0 the natural minimal case:

```python
>>> reals = float64.positive - float64.infinities
>>> reals.size
9218868437227405311
>>> float64.decode(float64.range(1.0, 2.0)[0])
1.0
>>> positives = float64.finite & float64.sign_clear
>>> [float64.decode(positives[at]) for at in range(2)]
[0.0, 5e-324]

```

Named sets: `positive`, `negative`, `nonnegative`, `zeros`, `subnormal`, `infinities`,
`nan`, `finite`, and the bit-level halves `sign_clear` / `sign_set`. Note a NaN has a
sign bit but is neither positive nor negative, so it is in `sign_clear` but not in
`positive`.

## Addresses

```python
>>> from bitpattern.codecs import ipv4
>>> spare = ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")
>>> [str(n) for n in ipv4.networks(spare)][:3]
['10.0.0.0/16', '10.2.0.0/15', '10.4.0.0/14']

```

`networks` reads the minimal CIDR cover straight off the diagram — route aggregation
for free. `ipv6` works the same way at 128 bits.

## Hypothesis

```bash
pip install 'bitpattern[hypothesis]'
```

```python
from hypothesis import given
from bitpattern.strategies import from_codec
from bitpattern.codecs import float64

@given(from_codec(float64, float64.finite))
def test_round_trips(value: float):
    ...
```

Drawing is by index, so a set of `2 ** 63` floats costs no more to sample than a set of
three, and shrinking the index shrinks the float.

## Patterns as output

Any set can be written back out as a pattern:

```python
>>> len(spare.pattern.branches)
8
>>> spare.pattern.branches[0]
'0000101000000000????????????????'

```

The pattern *language* describes one pattern. Unions are Python, and that is how they
print — as the expression that would rebuild them:

```python
>>> Pattern("0000") | Pattern("0001")
Pattern('000?')
>>> ~Pattern("00??")
Pattern('01??') | Pattern('1???')

```

Branches are derived from the diagram rather than remembered from the text, so patterns
are canonical: `Pattern("0000") | Pattern("0001")` *is* `Pattern("000?")`, and prints
that way.

A `Pattern` *is* an `IntSet` — it just also knows how to write itself down — so the two
mix freely, and operations on a pattern stay patterns:

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
| `bitpattern.codecs` | `float16/32/64`, `ipv4`, `ipv6`, `Codec` |
| `bitpattern.strategies` | `from_intset`, `from_codec` |
| `bitpattern.bdd` | the engine, if you're extending it |

MIT licensed. Requires Python 3.11+.
