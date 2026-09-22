"""`Pattern` parsing, and the sets patterns denote, against brute force."""

import itertools

import pytest

from bitpattern import IntSet, Pattern
from bitpattern.codecs import ipv4
from bitpattern.pattern import QUARTET, group

TARGET = "*1.*.*.0000.1111.?01?"


def group(bits: str) -> str:
    """Write a bit string as dotted quartets, leading group short if need be."""
    head = len(bits) % QUARTET or QUARTET
    return ".".join(
        [bits[:head]] + [bits[at:at + QUARTET] for at in range(head, len(bits), QUARTET)]
    )


def matches(bits: str, value: int) -> bool:
    """Whether `value` matches an expanded bit string, checked bit by bit."""
    width = len(bits)
    return all(
        char == "?" or int(char) == (value >> (width - 1 - at)) & 1
        for at, char in enumerate(bits)
    )


class TestTarget:
    """The pattern the interface was designed around."""

    pattern = Pattern(TARGET)

    def test_expansion(self):
        assert self.pattern.bits == "???1????????00001111?01?"

    def test_shape(self):
        assert self.pattern.width == 24
        assert self.pattern.free == 13

    def test_membership(self):
        assert len(self.pattern) == 1 << 13 == 8192
        assert self.pattern[0] == 1048818
        assert self.pattern[-1] == 16773371
        assert self.pattern[0] in self.pattern
        assert self.pattern[-1] in self.pattern
        assert 0 not in self.pattern

    def test_costs_one_node_per_pinned_bit(self):
        seen, stack = set(), [self.pattern.set.bdd]
        while stack:
            node = stack.pop()
            if getattr(node, "left", None) is not None and id(node) not in seen:
                seen.add(id(node))
                stack += [node.left, node.right]
        assert len(seen) == self.pattern.width - self.pattern.free == 11


class TestExpansion:
    @pytest.mark.parametrize(
        "text, bits",
        [
            ("*", "????"),
            ("*1", "???1"),
            ("1*", "1???"),
            ("1*1", "1??1"),
            ("??*?", "????"),
            ("0000", "0000"),
            ("?01?", "?01?"),
            ("*.*", "????????"),
            ("*1.0000", "???10000"),
            (TARGET, "???1????????00001111?01?"),
        ],
    )
    def test_glob_fills_its_own_quartet(self, text, bits):
        assert Pattern(text).bits == bits

    @pytest.mark.parametrize(
        "text, width",
        [("1", 1), ("01", 2), ("?01", 3), ("0000", 4), ("1.0000", 5), ("?01.1111", 7)],
    )
    def test_leading_group_may_be_short(self, text, width):
        assert Pattern(text).width == width

    @pytest.mark.parametrize(
        "text, members",
        [
            ("1", [1]),
            ("01", [1]),
            ("?1", [1, 3]),
            ("1.0000", [16]),
            ("?01.1111", [31, 95]),
            ("0.????", list(range(16))),
            ("1.????", list(range(16, 32))),
            ("?.0000", [0, 16]),
        ],
    )
    def test_short_leading_groups_denote_the_right_set(self, text, members):
        assert list(Pattern(text)) == members

    @pytest.mark.parametrize(
        "text",
        [
            "",  # empty pattern
            ".0000",  # empty leading group
            "0000.",  # empty trailing group
            "0000..0000",  # empty middle group
            "*1*",  # ambiguous width
            "00000",  # over a quartet
            "*00000",  # over a quartet, with a glob
            "0x0",  # not a bit
            "0000.01",  # only the leading group may be short
            "0000.1",
            "*.01",
            "00 00",  # whitespace inside a group
        ],
    )
    def test_rejected(self, text):
        with pytest.raises(ValueError):
            Pattern(text)

    def test_error_names_the_offending_group(self):
        with pytest.raises(ValueError, match=r"'0x0'"):
            Pattern("0x0.0000")


class TestBruteForce:
    """Every pattern over `01?` up to six bits wide, against direct matching.

    Widths 1, 2, 3, 5 and 6 are not multiples of four, so this is also where the
    short leading group gets its exercise.
    """

    @pytest.mark.parametrize("width", range(1, 7))
    def test_agrees_with_direct_matching(self, width):
        universe = range(1 << width)
        checked = 0
        for combination in itertools.product("01?", repeat=width):
            checked += 1
            bits = "".join(combination)
            pattern = Pattern(group(bits))
            expected = [value for value in universe if matches(bits, value)]

            assert pattern.bits == bits
            assert pattern.width == width
            assert len(pattern) == len(expected) == 1 << bits.count("?")
            assert list(pattern) == expected
            assert [pattern[at] for at in range(len(expected))] == expected
            assert [value in pattern for value in universe] == [
                value in expected for value in universe
            ]
        assert checked == 3 ** width


class TestCanonicalForm:
    @pytest.mark.parametrize(
        "text, canonical",
        [
            (TARGET, "???1.????.????.0000.1111.?01?"),
            ("*", "????"),
            ("1", "1"),
            ("1.0000", "1.0000"),
            ("?01.1111", "?01.1111"),
        ],
    )
    def test_str_is_the_expanded_spelling(self, text, canonical):
        assert str(Pattern(text)) == canonical
        assert repr(Pattern(text)) == f"Pattern({canonical!r})"

    @pytest.mark.parametrize("width", range(1, 7))
    def test_round_trips(self, width):
        for combination in itertools.product("01?", repeat=width):
            pattern = Pattern(group("".join(combination)))
            assert Pattern(str(pattern)) == pattern

    def test_spelling_does_not_affect_equality(self):
        assert Pattern("*") == Pattern("????")
        assert Pattern("*1") == Pattern("???1")
        assert Pattern("*") != Pattern("?.????")
        assert Pattern("*") != "????"

    def test_is_hashable(self):
        assert len({Pattern("*"), Pattern("????"), Pattern("*1")}) == 2


class TestSetOperations:
    narrow = Pattern("?1")  # {1, 3}
    wide = Pattern("00??")  # {0, 1, 2, 3}

    def test_across_widths(self):
        assert set(self.narrow & self.wide) == {1, 3}
        assert set(self.narrow | self.wide) == {0, 1, 2, 3}
        assert set(self.narrow - self.wide) == set()
        assert set(self.wide - self.narrow) == {0, 2}

    def test_result_takes_the_wider_universe(self):
        assert (self.narrow & self.wide).width == 4
        assert (self.narrow | self.wide).width == 4

    def test_accepts_plain_iterables(self):
        assert set(self.wide & [1, 2, 99]) == {1, 2}
        assert set(self.wide & IntSet([3])) == {3}

    def test_disjoint_patterns(self):
        assert set(Pattern("0000") & Pattern("1111")) == set()
        assert set(Pattern("0000") | Pattern("1111")) == {0, 15}


class TestUnions:
    """Unions are Python, not syntax -- but a `Pattern` still holds one."""

    @pytest.mark.parametrize(
        "text",
        ["0000 | 1111", "0000|1111", "0000 | 0001 | 0010", "*|*", "0000 |", "| 0000"],
    )
    def test_the_language_has_no_union(self, text):
        with pytest.raises(ValueError, match=r"combine them in Python"):
            Pattern(text)

    def test_the_error_shows_the_expression_to_use(self):
        with pytest.raises(ValueError, match=r"Pattern\('0000'\) \| Pattern\('1111'\)"):
            Pattern("0000 | 1111")

    @pytest.mark.parametrize(
        "built, members",
        [
            (lambda: Pattern("0000") | Pattern("1111"), [0, 15]),
            (lambda: Pattern("000?") | Pattern("1111"), [0, 1, 15]),
            (lambda: Pattern("0000") | Pattern("0001") | Pattern("0010"), [0, 1, 2]),
            (lambda: ~Pattern("00??"), list(range(4, 16))),
            (lambda: Pattern("0???") - Pattern("00??"), [4, 5, 6, 7]),
        ],
    )
    def test_the_operators_build_them(self, built, members):
        pattern = built()
        assert isinstance(pattern, Pattern)
        assert list(pattern) == members

    @pytest.mark.parametrize(
        "spelling, canonical",
        [
            (lambda: Pattern("0000") | Pattern("0001"), "000?"),
            (lambda: Pattern("0001") | Pattern("0000") | Pattern("0000"), "000?"),
            (lambda: Pattern("000?") | Pattern("0001"), "000?"),
            (lambda: Pattern("0???") | Pattern("1???"), "????"),
        ],
    )
    def test_is_canonical(self, spelling, canonical):
        """Branches come from the diagram, so spelling cannot survive."""
        built = spelling()
        assert built == Pattern(canonical)
        assert str(built) == str(Pattern(canonical))
        assert built.branches == Pattern(canonical).branches
        assert hash(built) == hash(Pattern(canonical))

    def test_repr_is_the_expression_that_rebuilds_it(self):
        pattern = ~Pattern("00??")
        assert repr(pattern) == "Pattern('01??') | Pattern('1???')"
        assert eval(repr(pattern)) == pattern

    def test_str_joins_the_branches(self):
        assert str(~Pattern("00??")) == "01?? | 1???"

    def test_a_long_union_is_summarised(self):
        pattern = (ipv4.cidr("10.0.0.0/8") - ipv4.cidr("10.1.0.0/16")).pattern
        assert repr(pattern) == "<Pattern: 8 branches, width=32, size=16711680>"

    @pytest.mark.parametrize("width", range(1, 5))
    def test_branches_are_well_formed(self, width):
        """Every subset of a small universe, written out and read back."""
        checked = 0
        for mask in range(1 << (1 << width)):
            members = {v for v in range(1 << width) if mask >> v & 1}
            pattern = IntSet(members, width).pattern
            checked += 1
            assert set(pattern) == members
            assert pattern.width == width
            for branch in pattern.branches:
                assert len(branch) == width
                assert set(branch) <= set("01?")
            # Branches are disjoint, so their sizes add up to the whole.
            assert sum(1 << b.count("?") for b in pattern.branches) == len(members)
        assert checked == 1 << (1 << width)

    @pytest.mark.parametrize("width", range(1, 4))
    def test_every_subset_rebuilds_from_its_branches(self, width):
        for mask in range(1 << (1 << width)):
            members = {v for v in range(1 << width) if mask >> v & 1}
            pattern = IntSet(members, width).pattern
            rebuilt = Pattern("", width=width)
            for branch in pattern.branches:
                rebuilt = rebuilt | Pattern(group(branch))
            assert rebuilt == pattern
            assert set(rebuilt) == members

    def test_free_bits_do_not_branch(self):
        """The trap: reusing the CIDR walk gave 32768 branches for one pattern."""
        odd = Pattern("????.????.????.???1")
        assert odd.branches == ("???????????????1",)
        assert odd.size == 1 << 15

    def test_is_closed_under_the_algebra(self):
        a, b = Pattern("00??"), Pattern("0?0?")
        for result in (a & b, a | b, a - b, ~a):
            assert isinstance(result, Pattern)
            assert result.width == 4
        assert set(a & b) == {0, 1}
        assert set(a | b) == {0, 1, 2, 3, 4, 5}
        assert set(a - b) == {2, 3}
        assert set(~a) == set(range(4, 16))

    def test_is_a_set(self):
        from collections.abc import Set

        assert isinstance(Pattern("00??"), Set)
        assert Pattern("00??") <= Pattern("0???")
        assert Pattern("0000").isdisjoint(Pattern("1111"))
        assert set(Pattern("00??")) == {0, 1, 2, 3}

    def test_empty_and_full(self):
        empty = Pattern("", width=4)
        assert empty.branches == () and empty.size == 0 and list(empty) == []
        assert Pattern("????").branches == ("????",)
        assert ~empty == Pattern("????")

    def test_the_empty_pattern_needs_a_width(self):
        with pytest.raises(ValueError, match="no width of its own"):
            Pattern("")

    def test_width_argument_must_agree(self):
        assert Pattern("0000", width=4) == Pattern("0000")
        with pytest.raises(ValueError):
            Pattern("0000", width=8)

    def test_bits_and_free_reject_unions(self):
        union = Pattern("0000") | Pattern("1111")
        for attribute in ("bits", "free"):
            with pytest.raises(ValueError, match="branches"):
                getattr(union, attribute)

    def test_pickles(self):
        import pickle

        pattern = ~Pattern("00??")
        restored = pickle.loads(pickle.dumps(pattern))
        assert type(restored) is Pattern and restored == pattern

    def test_intset_pattern_round_trips(self):
        source = IntSet([1, 4, 9, 16, 25], 8)
        assert set(source.pattern) == set(source)
        assert source.pattern == source
