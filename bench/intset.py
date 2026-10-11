"""IntSet workloads, timed against any implementation with the same API.

Each repetition builds fresh random inputs outside the timer, so caches keyed on
diagram identity can't answer a repeated question. Prints the best time per workload.

    python bench/intset.py py              # bitpattern.intset.IntSet
    python bench/intset.py rs              # the Rust spike, bitpattern_rs.IntSet
    python bench/intset.py py --only 's[i]'
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import importlib
import json
import operator
import pstats
import random
import time
from collections.abc import Callable, Iterator
from typing import Any

from bitpattern.intset import IntSet

REPEAT = 5

Setup = Callable[[random.Random], Any]
Run = Callable[[Any], object]


def implementation(name: str) -> type[IntSet]:
    """The IntSet class to time. The Rust spike's matches the Python one's API."""
    if name == "py":
        return IntSet
    return importlib.import_module("bitpattern_rs").IntSet


def timed(setup: Setup, run: Run, profiler: cProfile.Profile | None = None) -> float:
    """The best time of `run(setup(rng))` over REPEAT freshly seeded inputs. A
    `profiler` sees only the runs, not the setups."""
    best = float("inf")
    for seed in range(REPEAT):
        inputs = setup(random.Random(seed))
        if profiler:
            profiler.enable()
        start = time.perf_counter()
        run(inputs)
        best = min(best, time.perf_counter() - start)
        if profiler:
            profiler.disable()
    return best


def workloads(cls: type[IntSet]) -> Iterator[tuple[str, Setup, Run]]:
    def values(rng: random.Random, count: int, width: int) -> list[int]:
        return [rng.getrandbits(width) for _ in range(count)]

    def built(rng: random.Random, count: int, width: int) -> IntSet:
        return cls(values(rng, count, width), width=width)

    def ranges(rng: random.Random, count: int, width: int) -> list[tuple[int, int]]:
        pairs = (
            sorted((rng.getrandbits(width), rng.getrandbits(width)))
            for _ in range(count)
        )
        return [(low, high) for low, high in pairs]

    def some_ranges(rng: random.Random, count: int, width: int) -> list[IntSet]:
        return [
            cls.range(low, high, width=width) for low, high in ranges(rng, count, width)
        ]

    yield (
        "build 20k values (w=32)",
        lambda rng: values(rng, 20_000, 32),
        lambda v: cls(v, width=32),
    )
    yield (
        "2k ranges (w=64)",
        lambda rng: ranges(rng, 2_000, 64),
        lambda pairs: [cls.range(low, high, width=64) for low, high in pairs],
    )
    yield (
        "union of 2k ranges (w=32)",
        lambda rng: some_ranges(rng, 2_000, 32),
        lambda sets: functools.reduce(operator.or_, sets),
    )
    yield (
        "& | - ^ of 5k-member sets (w=24)",
        lambda rng: (built(rng, 5_000, 24), built(rng, 5_000, 24)),
        lambda ab: (ab[0] & ab[1], ab[0] | ab[1], ab[0] - ab[1], ab[0] ^ ab[1]),
    )
    yield (
        "~ of a 20k-member set (w=32)",
        lambda rng: built(rng, 20_000, 32),
        lambda s: ~s,
    )
    yield (
        "iterate 1M-member range",
        lambda rng: cls.range(rng.getrandbits(8), 1 << 20, width=21),
        lambda s: sum(1 for _ in s),
    )
    yield (
        "iterate 20k-member set (w=32)",
        lambda rng: built(rng, 20_000, 32),
        lambda s: sum(1 for _ in s),
    )
    yield (
        "100k s[i] on a 20k-member set",
        lambda rng: (
            built(rng, 20_000, 32),
            [rng.randrange(19_000) for _ in range(100_000)],
        ),
        lambda si: [si[0][i] for i in si[1]],
    )
    yield (
        "100k x in s on a 20k-member set",
        lambda rng: (built(rng, 20_000, 32), values(rng, 100_000, 32)),
        lambda sx: sum(x in sx[0] for x in sx[1]),
    )
    yield (
        "size of 1k fresh unions (w=64)",
        lambda rng: list(
            zip(some_ranges(rng, 1_000, 64), some_ranges(rng, 1_000, 64), strict=True)
        ),
        lambda pairs: [(a | b).size for a, b in pairs],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", choices=["py", "rs"])
    parser.add_argument(
        "--only", default="", help="run workloads whose name contains this"
    )
    parser.add_argument("--json", help="also write the times here")
    parser.add_argument(
        "--profile", action="store_true", help="print each run's hot spots"
    )
    args = parser.parse_args()
    results = {}
    for label, setup, run in workloads(implementation(args.implementation)):
        if args.only in label:
            profiler = cProfile.Profile() if args.profile else None
            results[label] = timed(setup, run, profiler)
            print(f"{label:40} {results[label] * 1000:9.1f} ms", flush=True)
            if profiler:
                pstats.Stats(profiler).sort_stats("tottime").print_stats(8)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
