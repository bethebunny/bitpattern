"""The published surface, and the reprs that have to survive a debugger."""

import time

import pytest

import bitpattern
from bitpattern import IntSet, Pattern
from bitpattern.bdd import ACCEPT, BDD, REJECT, node_count, render_count
from bitpattern.codecs import float16, float32, float64, ipv4, ipv6


class TestPublicSurface:
    def test_the_top_level_is_three_names(self):
        assert bitpattern.__all__ == ["IntSet", "Pattern", "__version__"]

    @pytest.mark.parametrize("name", ["ACCEPT", "REJECT", "BDD", "BDDLeaf", "BDDNode"])
    def test_the_engine_is_not_top_level(self, name):
        """These are the representation; exporting them would freeze it."""
        assert not hasattr(bitpattern, name)

    @pytest.mark.parametrize("name", ["Codec", "Float", "IP", "float64", "ipv4"])
    def test_codecs_are_not_top_level(self, name):
        assert not hasattr(bitpattern, name)

    def test_version(self):
        assert bitpattern.__version__.count(".") == 2

    def test_engine_names_live_in_bdd(self):
        import bitpattern.bdd

        assert bitpattern.bdd.__all__ == ["ACCEPT", "BDD", "BDDLeaf", "BDDNode", "REJECT"]

    @pytest.mark.parametrize("name", ["_align", "_coerce", "_extended", "_canonical"])
    def test_plumbing_is_private(self, name):
        assert hasattr(IntSet, name)
        assert not hasattr(IntSet, name.lstrip("_"))

    def test_is_typed(self):
        from pathlib import Path

        assert (Path(bitpattern.__file__).parent / "py.typed").exists()


class TestNodeRepr:
    """The dataclass default expanded sharing into a tree: 186MB at 43 nodes."""

    def test_leaves_repr_as_their_names(self):
        assert repr(ACCEPT) == "ACCEPT"
        assert repr(REJECT) == "REJECT"

    def test_a_deep_shared_dag_stays_bounded(self):
        a, b = ACCEPT, REJECT
        for depth in range(1, 23):
            a, b = BDD(depth, a, b), BDD(depth, b, a)
        start = time.perf_counter()
        text = repr(a)
        assert time.perf_counter() - start < 1.0
        assert len(text) < 100, text
        assert node_count(a) == 43

    def test_reports_the_shape(self):
        node = BDD(3, ACCEPT, REJECT)
        assert repr(node) == "<BDD bit=3, 1 nodes, 8 members>"

    def test_render_count(self):
        assert render_count(2**53) == "2**53"
        assert render_count(1024) == "2**10"
        assert render_count(512) == "512"  # below the threshold, plain is clearer
        assert render_count(8191) == "8191"


class TestIntSetRepr:
    def test_small_sets_round_trip(self):
        for group in (IntSet(), IntSet([1, 3, 8], 4), IntSet(range(10), 4)):
            assert eval(repr(group)) == group
            assert eval(repr(group)).width == group.width

    def test_empty(self):
        assert repr(IntSet([], 8)) == "IntSet([], width=8)"

    def test_large_sets_truncate(self):
        text = repr(IntSet.range(0, 10**6))
        assert text == "IntSet([0, 1, 2, 3, ..., 999998, 999999], size=1000000, width=20)"

    def test_a_set_len_cannot_measure(self):
        """`repr(float64.nan)` used to raise MemoryError."""
        start = time.perf_counter()
        text = repr(float64.nan)
        assert time.perf_counter() - start < 1.0
        assert len(text) < 250, text
        assert "size=9007199254740990" in text and "width=64" in text

    @pytest.mark.parametrize("source", ["nan", "finite", "subnormal", "sign_clear"])
    def test_every_codec_set_reprs_promptly(self, source):
        start = time.perf_counter()
        text = repr(getattr(float64, source))
        assert time.perf_counter() - start < 1.0
        assert len(text) < 250

    def test_shows_powers_of_two_readably(self):
        assert "size=2**24" in repr(ipv4.cidr("10.0.0.0/8"))


class TestPatternRepr:
    def test_single_branch_round_trips(self):
        for text in ("*1.*.*.0000.1111.?01?", "1", "?01.1111", "0000"):
            pattern = Pattern(text)
            assert eval(repr(pattern)) == pattern
            assert Pattern(str(pattern)) == pattern

    def test_canonical_spelling(self):
        assert repr(Pattern("*1.*.*.0000.1111.?01?")) == (
            "Pattern('???1.????.????.0000.1111.?01?')"
        )

    def test_a_union_reprs_as_the_expression_that_builds_it(self):
        pattern = Pattern("0000") | Pattern("1111")
        assert repr(pattern) == "Pattern('0000') | Pattern('1111')"
        assert eval(repr(pattern)) == pattern

    def test_many_branches_summarise(self):
        pattern = (ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")).pattern
        text = repr(pattern)
        assert text == "<Pattern: 8 branches, width=32, size=16711680>"
        assert len(text) < 300

    def test_the_empty_pattern_carries_its_width(self):
        empty = IntSet([], 4).pattern
        assert repr(empty) == "Pattern('', width=4)"
        assert eval(repr(empty)) == empty

    def test_a_scattered_set_stays_bounded(self):
        pattern = IntSet(range(0, 2**16, 3)).pattern
        assert len(pattern.branches) > 100
        assert len(repr(pattern)) < 300


class TestCodecRepr:
    @pytest.mark.parametrize(
        "codec, name",
        [(float16, "float16"), (float32, "float32"), (float64, "float64"),
         (ipv4, "ipv4"), (ipv6, "ipv6")],
    )
    def test_codecs_repr_as_their_names(self, codec, name):
        assert repr(codec) == name
        assert f"{codec}" == name


class TestReadme:
    """The README's examples are executable, so they cannot quietly go stale."""

    def test_examples_all_run(self):
        import doctest
        from pathlib import Path

        readme = Path(__file__).parent.parent / "README.md"
        failures, attempted = doctest.testfile(
            str(readme),
            module_relative=False,
            optionflags=doctest.IGNORE_EXCEPTION_DETAIL,
            verbose=False,
        )
        assert attempted >= 15
        assert failures == 0


class TestSubclassing:
    """`_coerce` used to be `cls(other)`, which corrupted subclass results."""

    class TextConstructed(IntSet):
        """A subclass whose constructor does not take an iterable, like Pattern."""

        __slots__ = ()

        def __init__(self, text: str):
            super().__init__([1, 2], 4)

    def test_operators_are_not_corrupted_by_the_constructor(self):
        group = self.TextConstructed("ignored")
        assert sorted(group & [1]) == [1]
        assert sorted(group - [2]) == [1]
        assert sorted(group | [8]) == [1, 2, 8]
        assert (group <= [1]) is False
        assert (group <= [1, 2, 3]) is True
        assert group.isdisjoint([7]) is True

    def test_mixin_results_are_plain_intsets(self):
        group = self.TextConstructed("ignored")
        assert sorted(group ^ IntSet([2, 3], 4)) == [1, 3]

    def test_results_keep_the_receiver_type(self):
        group = self.TextConstructed("ignored")
        for result in (group & [1], group | [8], group - [2], ~group, group.widen(8)):
            assert type(result) is self.TextConstructed


class TestPatternIsAnIntSet:
    def test_inherits_rather_than_forwards(self):
        assert issubclass(Pattern, IntSet)
        assert isinstance(Pattern("00??"), IntSet)

    def test_nothing_is_missing(self):
        """The forwarding it replaced had already drifted out of sync."""
        public = lambda cls: {n for n in dir(cls) if not n.startswith("_")}
        assert not public(IntSet) - public(Pattern)

    @pytest.mark.parametrize("name", ["widen", "range", "from_bdd", "pattern"])
    def test_the_previously_missing_methods(self, name):
        assert hasattr(Pattern("00??"), name)

    def test_operations_stay_patterns(self):
        pattern = Pattern("00??")
        for result in (pattern & [1], pattern | [8], pattern - [2], ~pattern,
                       pattern.widen(8), pattern.pattern):
            assert isinstance(result, Pattern)
            assert isinstance(result, IntSet)

    def test_widening_keeps_a_readable_spelling(self):
        assert repr(Pattern("00??").widen(8)) == "Pattern('0000.00??')"

    def test_set_gives_a_plain_intset(self):
        plain = Pattern("00??").set
        assert type(plain) is IntSet
        assert plain == Pattern("00??")

    def test_compares_and_hashes_with_intsets(self):
        assert Pattern("00??") == IntSet([0, 1, 2, 3], 4)
        assert hash(Pattern("00??")) == hash(IntSet([0, 1, 2, 3], 4))
        assert {Pattern("00??"): "a"}[IntSet([0, 1, 2, 3], 4)] == "a"

    def test_has_no_instance_dict(self):
        with pytest.raises(AttributeError):
            Pattern("00??").unexpected = 1
