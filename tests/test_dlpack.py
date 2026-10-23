"""Sets' members as DLPack tensors, read back by NumPy, and by hand for what NumPy
doesn't show."""

import ctypes
import gc
import math
import os
import random
from enum import IntEnum
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from bitpattern import Pattern, bdd
from bitpattern.bdd import BDD
from bitpattern.codecs import (
    DType,
    FloatFN,
    Int,
    Members,
    Tuple,
    UUIDCodec,
    bfloat16,
    float8_e4m3fn,
    float8_e5m2,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    int128,
    ipv4,
    ipv6,
    uint8,
    uint16,
    uint32,
    uint64,
    uuid128,
)
from bitpattern.intset import IntSet
from bitpattern.pattern import dotted


class Tensor(ctypes.Structure):
    """DLTensor."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("ndim", ctypes.c_int32),
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class Versioned(ctypes.Structure):
    """DLManagedTensorVersioned."""

    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("tensor", Tensor),
    ]


class Legacy(ctypes.Structure):
    """DLManagedTensor."""

    _fields_ = [
        ("tensor", Tensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


pointer = ctypes.pythonapi.PyCapsule_GetPointer
pointer.restype = ctypes.c_void_p
pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def words(tensor: Tensor) -> list[int]:
    """The tensor's values, as unsigned ints, row after row."""
    size, length = tensor.bits // 8, tensor.shape[0]
    if tensor.ndim == 2:
        length *= tensor.shape[1]
    data = ctypes.string_at(tensor.data, length * size) if length else b""
    chunks = (data[at : at + size] for at in range(0, len(data), size))
    return [int.from_bytes(chunk, "little") for chunk in chunks]


@pytest.mark.parametrize(
    "width, dtype",
    [
        (0, np.uint8),
        (1, np.uint8),
        (8, np.uint8),
        (9, np.uint16),
        (16, np.uint16),
        (17, np.uint32),
        (32, np.uint32),
        (33, np.uint64),
        (64, np.uint64),
    ],
)
def test_an_intset_exports_its_members_in_order(width, dtype):
    """As the narrowest unsigned ints that hold them."""
    rng = random.Random(width)
    group = IntSet([rng.getrandbits(width) for _ in range(500)], width=width)
    array = np.from_dlpack(group)
    assert array.dtype == dtype
    assert array.tolist() == list(group)


def test_runs_of_members_export_whole():
    group = IntSet.range(5, (1 << 20) + 7, width=24) | IntSet([1, 1 << 23], width=24)
    expected = [1, *range(5, (1 << 20) + 7), 1 << 23]
    assert np.from_dlpack(group).tolist() == expected


def patterns(width: int, seed: int) -> list[str]:
    """Bit patterns with up to 10 free bits, above, between and below fixed ones."""
    rng = random.Random(seed)
    texts = []
    for _ in range(40):
        free = set(rng.sample(range(width), min(width, rng.randint(0, 10))))
        bits = ("?" if at in free else rng.choice("01") for at in range(width))
        texts.append("".join(bits))
    return texts


@pytest.mark.parametrize("width", [1, 7, 8, 13, 16, 31, 33, 64])
def test_members_under_free_bits_export_in_order(width):
    """An export writes the members under free bits once, then copies them for the free
    bits' other values."""
    rng = random.Random(width)
    texts = patterns(width, width)
    for bits, other in zip(texts, reversed(texts), strict=True):
        group = Pattern(dotted(bits)) | Pattern(dotted(other))
        group |= IntSet([rng.getrandbits(width) for _ in range(20)], width=width)
        assert np.from_dlpack(group).tolist() == list(group), bits


@pytest.mark.parametrize("width", [2, 5, 8, 12, 33, 64])
def test_signed_values_under_a_free_sign_export_sign_extended(width):
    codec = Int(width, f"int{width}", signed=True)
    for bits in patterns(width, width):
        group = codec.pattern(dotted(bits)) | codec.pattern(dotted(bits[::-1]))
        assert np.from_dlpack(group).tolist() == list(group), bits


def test_a_versioned_tensor_is_a_new_aligned_1d_array_in_host_memory():
    group = IntSet([3, 300, 3000], width=12)
    capsule = group.__dlpack__(max_version=(1, 0))
    managed = Versioned.from_address(pointer(capsule, b"dltensor_versioned"))
    assert (managed.major, managed.minor) == (1, 0)
    assert managed.flags == 2  # copied, not read-only
    tensor = managed.tensor
    assert (tensor.device_type, tensor.device_id) == group.__dlpack_device__() == (1, 0)
    assert (tensor.ndim, tensor.code, tensor.bits, tensor.lanes) == (1, 1, 16, 1)
    assert (tensor.shape[0], tensor.strides[0], tensor.byte_offset) == (3, 1, 0)
    assert tensor.data % 256 == 0
    assert words(tensor) == [3, 300, 3000]


def test_every_export_is_aligned_to_256_bytes():
    """As DLPack asks. A small allocation can land on a 256-byte boundary by chance, so
    this tries many sizes."""
    for width in [8, 16, 32, 64]:
        for size in range(1, 200):
            capsule = IntSet(range(size), width=width).__dlpack__(max_version=(1, 0))
            managed = Versioned.from_address(pointer(capsule, b"dltensor_versioned"))
            assert managed.tensor.data % 256 == 0, (width, size)


@pytest.mark.parametrize("asked", [None, (0, 8)])
def test_a_consumer_from_before_dlpack_1_gets_the_legacy_capsule(asked):
    capsule = IntSet([3, 300], width=12).__dlpack__(max_version=asked)
    tensor = Legacy.from_address(pointer(capsule, b"dltensor")).tensor
    assert words(tensor) == [3, 300]


def test_an_empty_set_exports_an_empty_tensor():
    group = IntSet(width=16)
    # A capsule frees its tensor once it goes, so it's kept while the tensor is read.
    capsule = group.__dlpack__(max_version=(1, 0))
    tensor = Versioned.from_address(pointer(capsule, b"dltensor_versioned")).tensor
    assert tensor.shape[0] == 0 and tensor.data is None  # null, as DLPack asks
    array = np.from_dlpack(group)
    assert array.shape == (0,) and array.dtype == np.uint16


def test_an_exported_tensor_outlives_its_set():
    group = IntSet(range(0, 1000, 3), width=10)
    array = np.from_dlpack(group)
    del group
    gc.collect()
    bdd.collect()
    assert array.tolist() == list(range(0, 1000, 3))


def resident() -> int:
    """Bytes of this process in memory, from Linux's /proc."""
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


linux = pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="reads /proc")


@linux
def test_a_capsule_no_one_takes_frees_its_tensor():
    group = IntSet.range(0, 1 << 21, width=32)  # 8 MiB of words
    before = resident()
    for _ in range(40):
        group.__dlpack__(max_version=(1, 0))
        group.__dlpack__()
    assert resident() - before < 32 << 20  # where keeping them would be 640 MiB


@linux
def test_a_tensor_numpy_takes_is_freed_once():
    """NumPy renames the capsule as it takes the tensor, and calls the deleter itself."""
    group = IntSet.range(0, 1 << 21, width=32)
    before = resident()
    for _ in range(40):
        assert np.from_dlpack(group)[-1] == (1 << 21) - 1
    assert resident() - before < 32 << 20


def test_exports_only_go_to_host_memory_as_copies():
    group = IntSet([1, 2], width=8)
    assert np.from_dlpack(group, device="cpu", copy=True).tolist() == [1, 2]
    for device in [(2, 0), (1, 1)]:  # a GPU, and a second CPU
        with pytest.raises(BufferError):
            group.__dlpack__(dl_device=device)
    with pytest.raises(BufferError):
        group.__dlpack__(copy=False)
    with pytest.raises(ValueError, match="stream"):
        group.__dlpack__(stream=1)


@pytest.mark.parametrize("width", [65, 101, 128, 129, 200])
def test_an_intset_wider_than_64_bits_exports_a_row_of_bytes_per_member(width):
    """Most significant first, as int.to_bytes writes them, so rows sort like members."""
    rng = random.Random(width)
    ends = [0, (1 << width) - 1]
    group = IntSet([rng.getrandbits(width) for _ in range(300)] + ends, width=width)
    group |= IntSet.range(1 << (width - 1), (1 << (width - 1)) + 1000, width=width)
    array = np.from_dlpack(group)
    size = (width + 7) // 8
    assert array.dtype == np.uint8 and array.shape == (len(group), size)
    assert [row.tobytes() for row in array] == [m.to_bytes(size, "big") for m in group]


@pytest.mark.parametrize("width", [59, 64, 127, 128])
def test_a_set_too_big_for_memory_doesnt_export(width):
    """At 59 bits the words' 2**62 bytes fit an Int, so they reach the allocator."""
    with pytest.raises(MemoryError, match=f"{2**width} members"):
        IntSet.from_bdd(BDD.ACCEPT, width).__dlpack__()


def test_a_set_too_big_for_memory_names_a_huge_count_by_its_size():
    """Python won't write out an int this long."""
    with pytest.raises(MemoryError, match="an int of 1600001 bits members"):
        IntSet.from_bdd(BDD.ACCEPT, 1_600_000).__dlpack__()


@pytest.mark.parametrize(
    "codec", [int8, int16, int32, int64, uint8, uint16, uint32, uint64]
)
def test_an_int_set_exports_its_values_as_its_c_type(codec):
    """In the set's order, where the negatives come after the rest."""
    extremes = [codec.min, codec.min + 1, -1, 0, 1, codec.max]
    group = codec.set(value for value in extremes if codec.min <= value <= codec.max)
    array = np.from_dlpack(group)
    assert array.dtype == np.dtype(codec.name)
    assert array.tolist() == list(group)
    assert group.__dlpack_device__() == (1, 0)


def test_a_narrow_unsigned_int_set_exports_as_the_next_c_type_up():
    group = Int(12, "uint12", signed=False).set([0, 4095])
    array = np.from_dlpack(group)
    assert array.dtype == np.uint16
    assert array.tolist() == [0, 4095]


@pytest.mark.parametrize("codec", [float16, float32, float64])
def test_a_float_set_exports_its_values_bit_for_bit(codec):
    """NaNs and -0.0 too, in the set's order."""
    values = [0.0, -0.0, 1.5, -2.25, math.inf, -math.inf]
    group = codec.set(values) | codec.nan[:2] | codec.nan[-2:]
    array = np.from_dlpack(group)
    assert array.dtype == np.dtype(codec.name)
    assert array.view(f"u{codec.width // 8}").tolist() == list(group.storage)


def test_a_bfloat16_set_exports_bfloat16s():
    """NumPy has no bfloat16, but PyTorch and JAX do."""
    group = bfloat16.set([1.0, -2.0, math.inf])
    capsule = group.__dlpack__(max_version=(1, 0))
    tensor = Versioned.from_address(pointer(capsule, b"dltensor_versioned")).tensor
    assert (tensor.code, tensor.bits) == (4, 16)
    assert words(tensor) == list(group.storage)


def test_a_narrow_signed_int_set_exports_sign_extended():
    group = Int(12, "int12", signed=True).set([-2048, -1, 0, 5, 2047])
    array = np.from_dlpack(group)
    assert array.dtype == np.int16
    assert array.tolist() == list(group)


@pytest.mark.parametrize("codec, code", [(float8_e4m3fn, 10), (float8_e5m2, 12)])
def test_a_float8_set_exports_to_any_consumer(codec, code):
    """Its codes came in DLPack 1.1. A consumer checks the version it gets, so one that
    asks for 1.0, as torch does, gets them too."""
    group = codec.set([1.0, -2.0]) | codec.nan
    for asked in [(1, 0), (1, 1)]:
        capsule = group.__dlpack__(max_version=asked)
        managed = Versioned.from_address(pointer(capsule, b"dltensor_versioned"))
        assert (managed.major, managed.minor) == (1, 1)
        assert (managed.tensor.code, managed.tensor.bits) == (code, 8)
        assert words(managed.tensor) == list(group.storage)
    capsule = group.__dlpack__()
    tensor = Legacy.from_address(pointer(capsule, b"dltensor")).tensor
    assert (tensor.code, tensor.bits) == (code, 8)


def test_an_fn_float_set_exports_its_encodings_unless_its_e4m3fn():
    """Its format names a C type with infinities where it has numbers."""
    codec = FloatFN(16, "float16fn", exponent=5, format=">e")
    group = codec.set([1.0, 65536.0])  # 65536.0 has float16's infinity's bits
    array = np.from_dlpack(group)
    assert array.dtype == np.uint16 and array.tolist() == list(group.storage)


def test_other_codecs_export_their_encodings_as_intsets_do():
    addresses = ipv4.set([IPv4Address(1), IPv4Address(0xC0A80001)])
    array = np.from_dlpack(addresses)
    assert array.dtype == np.uint32 and array.tolist() == [1, 0xC0A80001]
    pairs = Tuple(uint8, uint16).set([(1, 2), (3, 4)])
    array = np.from_dlpack(pairs)
    assert array.dtype == np.uint32 and array.tolist() == list(pairs.storage)


class Level(IntEnum):
    LOW = 10
    HIGH = 20


def test_an_int_enum_set_exports_its_members_indices():
    array = np.from_dlpack(Members(Level).set([Level.LOW, Level.HIGH]))
    assert array.tolist() == [0, 1]


def test_wide_values_export_as_their_bytes():
    """For IPv6 addresses and UUIDs, those are their packed forms, and a product's are
    its fields' bytes one after another."""
    addresses = ipv6.set([IPv6Address(1), IPv6Address("2001:db8::1")])
    packed = [row.tobytes() for row in np.from_dlpack(addresses)]
    assert packed == [address.packed for address in addresses]
    ids = uuid128.set([UUID(int=5), UUID("12345678-1234-5678-1234-567812345678")])
    rows = [row.tobytes() for row in np.from_dlpack(ids)]
    assert rows == [value.bytes for value in ids]
    numbers = int128.set([-1, 0, 5])  # as two's complement
    rows = [row.tobytes() for row in np.from_dlpack(numbers)]
    assert [int.from_bytes(row, signed=True) for row in rows] == list(numbers)
    services = Tuple(ipv6, uint16).set([(IPv6Address(1), 80)])
    rows = [row.tobytes() for row in np.from_dlpack(services)]
    assert rows == [IPv6Address(1).packed + (80).to_bytes(2)]


class Halves(UUIDCodec):
    """UUIDs as their two halves, most significant first."""

    def dtype(self) -> DType:
        return DType.UINT64


def test_a_codec_chooses_the_type_its_sets_export_as():
    ids = Halves(128, "halves").set([UUID(int=(1 << 64) + 2), UUID(int=3)])
    array = np.from_dlpack(ids)
    assert array.dtype == np.uint64 and array.tolist() == [[0, 3], [1, 2]]


@pytest.mark.parametrize("wrong", [DType.FLOAT32, DType.INT8])
def test_a_type_too_narrow_or_of_another_size_is_a_value_error(wrong):
    class Wrong(Int):
        def dtype(self) -> DType:
            return wrong

    with pytest.raises(ValueError, match="can't hold 16-bit values"):
        np.from_dlpack(Wrong(16, "wrong", signed=False).set([1]))


@pytest.mark.parametrize("bits", [0, 12, 128])
def test_words_are_8_16_32_or_64_bits(bits):
    node = IntSet([1], width=8).bdd
    with pytest.raises(ValueError, match=f"not {bits}$"):
        bdd.tensor(node, width=8, code=1, bits=bits, version=None)
