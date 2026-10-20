"""Sets' members as DLPack tensors, read back by MAX. It asks for no DLPack version, so
it takes the legacy capsule."""

import gc
import os
import random
from ipaddress import IPv6Address
from pathlib import Path
from uuid import UUID

import pytest
from max.driver import Buffer
from max.dtype import DType

from bitpattern import bdd
from bitpattern.codecs import (
    BDDSet,
    Int,
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
    ipv6,
    uint8,
    uint16,
    uint32,
    uint64,
    uuid128,
)
from bitpattern.intset import IntSet

UNSIGNED = {1: DType.uint8, 2: DType.uint16, 4: DType.uint32, 8: DType.uint64}


def words(buffer: Buffer) -> list[int]:
    """The buffer's words as unsigned ints, row after row."""
    unsigned = buffer.view(UNSIGNED[buffer.dtype.size_in_bytes], buffer.shape)
    return unsigned.to_numpy().reshape(-1).tolist()


@pytest.mark.parametrize(
    "width, dtype",
    [
        (0, DType.uint8),
        (1, DType.uint8),
        (8, DType.uint8),
        (9, DType.uint16),
        (16, DType.uint16),
        (17, DType.uint32),
        (32, DType.uint32),
        (33, DType.uint64),
        (64, DType.uint64),
    ],
)
def test_an_intset_reads_back_as_its_members(width, dtype):
    rng = random.Random(width)
    group = IntSet([rng.getrandbits(width) for _ in range(500)], width=width)
    buffer = Buffer.from_dlpack(group)
    assert buffer.dtype == dtype
    assert buffer.to_numpy().tolist() == list(group)


@pytest.mark.parametrize(
    "codec, dtype",
    [
        (int8, DType.int8),
        (int16, DType.int16),
        (int32, DType.int32),
        (int64, DType.int64),
        (uint8, DType.uint8),
        (uint16, DType.uint16),
        (uint32, DType.uint32),
        (uint64, DType.uint64),
        (Int(12, "int12", signed=True), DType.int16),  # sign-extended
    ],
)
def test_an_int_set_reads_back_as_its_c_type(codec, dtype):
    rng = random.Random(codec.width)
    bits = [rng.getrandbits(codec.width) for _ in range(500)]
    group = BDDSet(codec, IntSet(bits, width=codec.width))
    buffer = Buffer.from_dlpack(group)
    assert buffer.dtype == dtype
    assert buffer.to_numpy().tolist() == list(group)


@pytest.mark.parametrize(
    "codec, dtype",
    [
        (float16, DType.float16),
        (float32, DType.float32),
        (float64, DType.float64),
        (bfloat16, DType.bfloat16),
        (float8_e4m3fn, DType.float8_e4m3fn),
        (float8_e5m2, DType.float8_e5m2),
    ],
)
def test_a_float_set_reads_back_bit_for_bit(codec, dtype):
    """Its bits as they are: NaNs, zeros of either sign and subnormals too."""
    rng = random.Random(codec.width)
    width = codec.width
    edges = [0, 1, 1 << (width - 1), (1 << width) - 1, (1 << (width - 1)) - 1]
    bits = edges + [rng.getrandbits(width) for _ in range(2000)]
    group = BDDSet(codec, IntSet(bits, width=width)) & codec.all
    buffer = Buffer.from_dlpack(group)
    assert buffer.dtype == dtype
    assert words(buffer) == list(group.storage)


def test_wide_values_read_back_as_rows_of_their_bytes():
    addresses = ipv6.set([IPv6Address(1), IPv6Address("2001:db8::1")])
    rows = Buffer.from_dlpack(addresses).to_numpy()
    assert [row.tobytes() for row in rows] == [value.packed for value in addresses]
    ids = uuid128.set([UUID(int=5), UUID(int=(1 << 127) + 9)])
    rows = Buffer.from_dlpack(ids).to_numpy()
    assert [row.tobytes() for row in rows] == [value.bytes for value in ids]
    numbers = int128.set([-1, 0, 5])  # as two's complement
    rows = Buffer.from_dlpack(numbers).to_numpy()
    assert [int.from_bytes(row.tobytes(), signed=True) for row in rows] == list(numbers)
    buffer = Buffer.from_dlpack(IntSet([3, 1 << 199], width=200))
    assert buffer.dtype == DType.uint8 and buffer.shape == (2, 25)
    assert [int.from_bytes(row.tobytes()) for row in buffer.to_numpy()] == [3, 1 << 199]


@pytest.mark.parametrize("width", [8, 64, 65])
def test_an_empty_set_reads_back_empty(width):
    assert Buffer.from_dlpack(IntSet(width=width)).num_elements == 0


def test_a_buffer_outlives_its_set():
    group = IntSet(range(0, 1000, 3), width=10)
    buffer = Buffer.from_dlpack(group)
    del group
    gc.collect()
    bdd.collect()
    assert buffer.to_numpy().tolist() == list(range(0, 1000, 3))


def resident() -> int:
    """Bytes of this process in memory, from Linux's /proc."""
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


@pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="reads /proc")
def test_a_tensor_max_takes_is_freed_once():
    """MAX calls the legacy capsule's deleter itself."""
    group = IntSet.range(0, 1 << 21, width=32)  # 8 MiB of words
    before = resident()
    for _ in range(40):
        assert Buffer.from_dlpack(group).to_numpy()[-1] == (1 << 21) - 1
    assert resident() - before < 32 << 20  # where keeping them would be 320 MiB
