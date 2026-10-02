"""The public names, reprs, and the README."""

import doctest
import importlib
import random
from pathlib import Path

import pytest

import bitpattern
from bitpattern import IntSet, Pattern, bdd
from bitpattern.bdd import BDD, node_count, render_count
from bitpattern.codecs import BDDSet, Codec, float16, float32, float64, ipv4, ipv6

README = Path(__file__).parent.parent / "README.md"


class TextConstructed(IntSet):
    """A subclass whose constructor doesn't take an iterable, like Pattern."""

    __slots__ = ()

    def __init__(self, text: str) -> None:
        super().__init__([1, 2], width=4)


def test_top_level_names():
    exported = {name: getattr(bitpattern, name) for name in bitpattern.__all__}
    assert exported == {
        "BDD": BDD,
        "BDDSet": BDDSet,
        "Codec": Codec,
        "IntSet": IntSet,
        "Pattern": Pattern,
    }


@pytest.mark.parametrize("name", ["float64", "ipv4"])
def test_codecs_arent_top_level(name):
    assert not hasattr(bitpattern, name)


def test_engine_names_live_in_bdd():
    assert bdd.__all__ == ["BDD"]


def test_version():
    assert bitpattern.__version__.count(".") == 2


def test_is_typed():
    assert (Path(bitpattern.__file__).parent / "py.typed").exists()


def test_readme_examples_run():
    random.seed(0)  # the README draws random members
    failures, attempted = doctest.testfile(
        str(README),
        module_relative=False,
        optionflags=doctest.IGNORE_EXCEPTION_DETAIL,
    )
    assert attempted >= 15
    assert not failures


@pytest.mark.parametrize(
    "module",
    [
        "bitpattern.bdd",
        "bitpattern.intset",
        "bitpattern.pattern",
        "bitpattern.codecs.codec",
    ],
)
def test_docstring_examples_run(module):
    failures, attempted = doctest.testmod(importlib.import_module(module))
    assert attempted
    assert not failures


def test_leaves_repr_as_their_names():
    assert repr(BDD.ACCEPT) == "BDD.ACCEPT"
    assert repr(BDD.REJECT) == "BDD.REJECT"
    assert eval(repr(BDD.ACCEPT)) is BDD.ACCEPT


def test_a_deep_shared_dag_reprs_briefly():
    """Diagrams share nodes, so written out as a tree they'd be exponentially big."""
    a, b = BDD.ACCEPT, BDD.REJECT
    for depth in range(1, 23):
        a, b = BDD(depth, a, b), BDD(depth, b, a)
    assert len(repr(a)) < 100
    assert node_count(a) == 43


def test_node_repr_summarizes_the_shape():
    assert repr(BDD(3, BDD.ACCEPT, BDD.REJECT)) == "<BDD bit=3, 1 nodes, 8 members>"


def test_render_count():
    assert render_count(2**53) == "2**53"
    assert render_count(1024) == "2**10"
    assert render_count(512) == "512"  # small enough to read as is
    assert render_count(8191) == "8191"


def test_small_intset_reprs_round_trip():
    for group in (IntSet(), IntSet([1, 3, 8], width=4), IntSet(range(10), width=4)):
        assert eval(repr(group)) == group
        assert eval(repr(group)).width == group.width


def test_empty_intset_repr():
    assert repr(IntSet([], width=8)) == "IntSet([], width=8)"


def test_big_intset_reprs_are_truncated():
    text = repr(IntSet.range(0, 10**6))
    assert text == "IntSet([0, 1, 2, 3, ..., 999998, 999999], size=1000000, width=20)"


@pytest.mark.parametrize("name", ["nan", "finite", "subnormal", "sign_clear"])
def test_sets_len_cant_measure_repr_briefly(name):
    assert len(repr(getattr(float64, name))) < 250


def test_float64_nan_repr():
    text = repr(float64.nan)
    assert text.startswith("BDDSet(float64, [nan, ") and "size=9007199254740990" in text


def test_reprs_show_powers_of_two_readably():
    assert "size=2**24" in repr(ipv4.cidr("10.0.0.0/8"))


def test_single_branch_patterns_round_trip():
    for text in ("*1.*.*.0000.1111.?01?", "1", "?01.1111", "0000"):
        pattern = Pattern(text)
        assert eval(repr(pattern)) == pattern
        assert Pattern(str(pattern)) == pattern


def test_pattern_repr_is_the_canonical_spelling():
    assert repr(Pattern("*1.*.*.0000.1111.?01?")) == (
        "Pattern('???1.????.????.0000.1111.?01?')"
    )


def test_unions_repr_as_the_expression_that_builds_them():
    pattern = Pattern("0000") | Pattern("1111")
    assert repr(pattern) == "Pattern('0000') | Pattern('1111')"
    assert eval(repr(pattern)) == pattern


def test_lots_of_branches_are_summarized():
    spare = (ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")).storage
    pattern = Pattern.from_bdd(spare.bdd, spare.width)
    assert repr(pattern) == "<Pattern: 8 branches, width=32, size=16711680>"


def test_the_empty_pattern_is_the_complement_of_everything():
    empty = Pattern.from_bdd(BDD.REJECT, 5)
    assert repr(empty) == "~Pattern('?.????')"
    assert eval(repr(empty)) == empty and eval(repr(empty)).width == 5


def test_exponentially_many_branches_repr_briefly():
    even, odd = BDD.ACCEPT, BDD.REJECT  # popcount parity: 128 nodes, 2**63 branches
    for bit in range(64):
        even, odd = BDD(bit, even, odd), BDD(bit, odd, even)
    assert repr(Pattern.from_bdd(even, 64)) == (
        "<Pattern: 2**63 branches, width=64, size=2**63>"
    )


def test_a_scattered_set_reprs_briefly():
    scattered = IntSet(range(0, 2**16, 3))
    pattern = Pattern.from_bdd(scattered.bdd, scattered.width)
    assert len(pattern.branches) > 100
    assert len(repr(pattern)) < 300


@pytest.mark.parametrize(
    "codec, name",
    [
        (float16, "float16"),
        (float32, "float32"),
        (float64, "float64"),
        (ipv4, "ipv4"),
        (ipv6, "ipv6"),
    ],
)
def test_codecs_repr_as_their_names(codec, name):
    assert repr(codec) == name
    assert str(codec) == name


def test_subclass_operators_work():
    """Operators can't build their results with the subclass's constructor."""
    group = TextConstructed("ignored")
    assert sorted(group & [1]) == [1]
    assert sorted(group - [2]) == [1]
    assert sorted(group | [8]) == [1, 2, 8]
    assert (group <= IntSet([1])) is False
    assert (group <= IntSet([1, 2, 3])) is True
    assert group.isdisjoint([7]) is True


def test_results_keep_the_subclass():
    group = TextConstructed("ignored")
    for result in (
        group & [1],
        group | [8],
        group - [2],
        group ^ [3],
        ~group,
        group.widen(8),
        group[:1],
    ):
        assert type(result) is TextConstructed
    assert sorted(group ^ IntSet([2, 3], width=4)) == [1, 3]


def test_pattern_is_an_intset():
    assert issubclass(Pattern, IntSet)
    assert isinstance(Pattern("00??"), IntSet)


def test_pattern_has_everything_intset_has():
    def public(cls: type) -> set[str]:
        return {name for name in dir(cls) if not name.startswith("_")}

    assert public(IntSet) <= public(Pattern)


def test_operations_on_patterns_stay_patterns():
    pattern = Pattern("00??")
    for result in (
        pattern & [1],
        pattern | [8],
        pattern - [2],
        ~pattern,
        pattern.widen(8),
        pattern[1:3],
    ):
        assert isinstance(result, Pattern)


def test_widening_keeps_a_readable_spelling():
    assert repr(Pattern("00??").widen(8)) == "Pattern('0000.00??')"


def test_patterns_compare_and_hash_with_intsets():
    assert Pattern("00??") == IntSet([0, 1, 2, 3], width=4)
    assert hash(Pattern("00??")) == hash(IntSet([0, 1, 2, 3], width=4))
    table: dict[IntSet, str] = {Pattern("00??"): "a"}
    assert table[IntSet([0, 1, 2, 3], width=4)] == "a"


def test_patterns_have_no_instance_dict():
    assert not hasattr(Pattern("00??"), "__dict__")
