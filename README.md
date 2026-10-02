# bitpattern

> [!NOTE]
> ```bash
> pip install bitpattern
> pip install 'bitpattern[hypothesis]'  # with the hypothesis strategies
> ```

bitpattern is a library for efficiently describing and working with very large
sets of structured data. It allows
- expressing structured bit patterns via a glob-like syntax
- composing these pattern sets via normal set operations
- very efficient sampling and counting of sets much larger than would
  ordinarily fit in memory

bitpattern uses a data structure called a Binary Decision Diagram to encode
extremely large sets. Whereas set operations are typically described in terms
of the size of the set, bitpattern sets support most operations in O(#bits) of
the _largest member_ of the set. #bits is called the `width` of the set.

```python
>>> from bitpattern import Pattern
>>> p = Pattern("*1.*.*.0000.1111.?01?")
>>> p
Pattern('???1.????.????.0000.1111.?01?')
>>> p.width
24
>>> p.size
8192
>>> p[0]
1048818
>>> 0x1230FA in p
True
>>> import random
>>> random.sample(p, 3)
[13799675, 14393594, 1724667]

```

The pattern syntax follows normal glob rules. Patterns are expressed as quartets
of 4 bits, most significant to least significant, with 0 and 1 representing a fixed
bit, ? can be either, and * is shorthand for multiple ?.

Patterns are sets, and work with normal set operations.

```python
>>> Pattern("0000") | Pattern("0001")
Pattern('000?')
>>> ~Pattern("00??")
Pattern('01??') | Pattern('1???')
>>> Pattern("0???") & Pattern("??00")
Pattern('0?00')
>>> Pattern("0???") - Pattern("00??")
Pattern('01??')
>>> Pattern("000?") <= Pattern("00??")
True

```

## Patterns for sets of structured data

Patterns start to really shine when expressing structured sets. Consider for example
64 bit floats. How many finite normal floats are there? Can we sample them directly?
These sets are huge and noncontiguous, so you can't express them in traditional data structures.

bitpattern allows equipping a bit pattern set with a structured type via a Codec
that encodes/decodes. This allows them to be used directly with structured types like floats.

```python
>>> from bitpattern.codecs import float64
>>> supported = float64.finite - float64.subnormal
>>> supported.size
18428729675200069634
>>> list(supported[:5])
[0.0, 2.2250738585072014e-308, 2.225073858507202e-308, 2.2250738585072024e-308, 2.225073858507203e-308]
>>> nans = float64.pattern("?111.1111.1111.0*.*.*.*.*.*.*.*.*.*.*.*.*")
>>> 0.25 in supported
True
>>> 0.25 in nans
False
>>> supported.choice()
-7.966640143021799e+266

```

These are `BDDSet`s, which are sets of values backed by the `IntSet` of their bits,
eg. `supported.storage`.

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

## IP addresses

`bitpattern.codecs` also has `ipv4` and `ipv6`, and `networks()` finds the fewest CIDR
blocks that make up a set of addresses.

```python
>>> from bitpattern.codecs import ipv4
>>> spare = ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")
>>> [str(n) for n in ipv4.networks(spare)][:3]
['10.0.0.0/16', '10.2.0.0/15', '10.4.0.0/14']

```

## IntSet

`IntSet` is the backing abstraction for Patterns and codecs, and is provided directly.
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
`|a|` is at most `n * w`, and is usually much smaller. Notably, any set that is expressible
via the Pattern language has `|a| <= 2w`. For most use cases `w` is a constant and may be read as `O(1)`.

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
