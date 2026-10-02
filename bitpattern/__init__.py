"""Sets of integers, described by their bits and backed by binary decision diagrams.

    >>> from bitpattern import Pattern
    >>> Pattern("*1.*.*.0000.1111.?01?").size
    8192

Codecs for floats and IP addresses are in `bitpattern.codecs`, hypothesis
strategies are in `bitpattern.strategies`, and the diagrams themselves are in
`bitpattern.bdd`.
"""

from .bdd import BDD
from .codecs import BDDSet, Codec
from .intset import IntSet
from .pattern import Pattern

__version__ = "0.1.0"

__all__ = ["BDD", "BDDSet", "Codec", "IntSet", "Pattern"]
