"""Bit patterns and integer sets backed by binary decision diagrams.

    >>> from bitpattern import Pattern
    >>> Pattern("*1.*.*.0000.1111.?01?").size
    8192

The engine lives in `bitpattern.bdd`, value codecs in `bitpattern.codecs`, and
hypothesis strategies in `bitpattern.strategies`.
"""

from .pattern import Pattern
from .sets import IntSet

__version__ = "0.1.0"

__all__ = ["IntSet", "Pattern"]
