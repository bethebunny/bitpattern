"""The UUID codec, checked against the uuid module and RFC 9562's examples."""

import itertools
import random
import uuid

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitpattern.bdd import node_count
from bitpattern.codecs.uuid import uuid128

VERSIONS = range(1, 9)

# The examples in RFC 9562's appendices. It has none for version 2.
RFC_EXAMPLES = [
    ("c232ab00-9414-11ec-b3c8-9f6bdeced846", 1),
    ("5df41881-3aed-3515-88a7-2f4a814cf09e", 3),
    ("919108f7-52d1-4320-9bac-f847db4148a8", 4),
    ("2ed6657d-e927-568b-95e1-2665a8aea6a2", 5),
    ("1ec9414c-232a-6b00-b3c8-9f6bdeced846", 6),
    ("017f22e2-79b0-7cc3-98c4-dc0c0c07398f", 7),
    ("2489e9ad-2ee2-8e00-8ec9-32d5f69181c0", 8),
    ("5c146b14-3c52-8afd-938a-375d0df1fbf6", 8),
]


def with_digits(text: str, version: str, variant: str) -> uuid.UUID:
    """`text` with M and N replaced, in `xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx`."""
    return uuid.UUID(text[:14] + version + text[15:19] + variant + text[20:])


# Every version and variant digit, with the other bits all clear, all set or mixed.
DIGITS = [
    with_digits(text, version, variant)
    for text in (
        "00000000-0000-0000-0000-000000000000",
        "ffffffff-ffff-ffff-ffff-ffffffffffff",
        "919108f7-52d1-4320-9bac-f847db4148a8",
    )
    for version, variant in itertools.product("0123456789abcdef", repeat=2)
]


@given(st.integers(0, 2**128 - 1))
@settings(deadline=None)
def test_uuids_round_trip_through_their_bits(bits):
    value = uuid128.decode(bits)
    assert value == uuid.UUID(int=bits)
    assert uuid128.encode(value) == bits
    assert uuid128.encode(str(value)) == bits


@pytest.mark.parametrize(
    "text",
    [
        "919108f7-52d1-4320-9bac-f847db4148a8",
        "919108F7-52D1-4320-9BAC-F847DB4148A8",
        "919108f752d143209bacf847db4148a8",
        "{919108f7-52d1-4320-9bac-f847db4148a8}",
        "urn:uuid:919108f7-52d1-4320-9bac-f847db4148a8",
    ],
)
def test_encode_takes_the_strings_uuid_takes(text):
    assert uuid128.encode(text) == 0x919108F752D143209BACF847DB4148A8


@pytest.mark.parametrize(
    "value",
    [
        "919108f7-52d1-4320-9bac",
        "919108f7-52d1-4320-9bac-f847db4148ag",
        "not a uuid",
        0x919108F752D143209BACF847DB4148A8,  # UUID() only takes ints by keyword
        bytes(16),
        None,
    ],
)
def test_encode_refuses_anything_else(value):
    with pytest.raises(ValueError):
        uuid128.encode(value)
    assert value not in uuid128.all


@pytest.mark.parametrize("text, n", RFC_EXAMPLES)
def test_rfc_examples_are_only_in_their_version(text, n):
    assert [m for m in VERSIONS if text in uuid128.version(m)] == [n]


def test_version_4_holds_exactly_what_uuid4_makes():
    # uuid4 randomizes all but the 6 version and variant bits, so it can make any
    # of the 2**122 UUIDs that have them.
    assert uuid128.version(4).size == 2**122
    assert all(uuid.uuid4() in uuid128.version(4) for _ in range(5000))


def test_rfc9562_is_the_uuid_modules_rfc_variant():
    for value in DIGITS:
        assert (value in uuid128.rfc9562) == (value.variant == uuid.RFC_4122)


@pytest.mark.parametrize("n", VERSIONS)
def test_version_is_the_uuid_modules_version(n):
    for value in DIGITS:
        assert (value in uuid128.version(n)) == (value.version == n)


def test_version_4_is_its_digits_as_a_pattern():
    """`xxxxxxxx-xxxx-4xxx-Nxxx-xxxxxxxxxxxx`, where N is 8, 9, a or b."""
    pattern = "*.*.*.*.*.*.*.*.*.*.*.*.0100.*.*.*.10??" + ".*" * 15
    assert uuid128.version(4) == uuid128.pattern(pattern)


def test_rfc_sets_are_a_node_per_fixed_bit():
    assert node_count(uuid128.rfc9562.storage.bdd) == 2
    assert {node_count(uuid128.version(n).storage.bdd) for n in VERSIONS} == {6}


@pytest.mark.parametrize("n", [0, 9, 15])
def test_versions_go_from_1_to_8(n):
    with pytest.raises(ValueError):
        uuid128.version(n)


@pytest.mark.parametrize("n", VERSIONS)
def test_version_choice_makes_uuids_of_that_version(n):
    rng = random.Random(n)
    for _ in range(200):
        value = uuid128.version(n).choice(rng=rng)
        assert isinstance(value, uuid.UUID)
        assert value.version == n and value.variant == uuid.RFC_4122


def test_min_and_max_are_the_nil_and_max_uuids():
    assert uuid128.min == uuid.UUID("00000000-0000-0000-0000-000000000000")
    assert uuid128.max == uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    assert uuid128.min not in uuid128.rfc9562
    assert uuid128.max not in uuid128.rfc9562


def test_uuid_ranges_go_in_uuid_order():
    start = uuid.UUID("12345678-1234-4234-8234-123456789abc")
    assert uuid128.range(start)[0] == start
    assert uuid128.range(start).size == 2**128 - start.int
    assert uuid128.range(stop=start).size == start.int


def test_version_4_runs_from_its_smallest_uuid_to_its_largest():
    assert uuid128.version(4)[0] == uuid.UUID("00000000-0000-4000-8000-000000000000")
    assert uuid128.version(4)[-1] == uuid.UUID("ffffffff-ffff-4fff-bfff-ffffffffffff")


@given(st.lists(st.uuids(), max_size=10))
@settings(deadline=None)
def test_uuids_sort_the_way_the_uuid_module_does(values):
    assert list(uuid128.set(values)) == sorted(set(values))
