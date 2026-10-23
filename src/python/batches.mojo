"""Python iterators over a set's members, blocks and paths, a batch at a time."""

from std.ffi import c_int
from std.memory.alloc import unsafe_alloc
from std.python._cpython import PyObjectPtr
from std.sys.info import size_of
from std.utils import Variant

from arena import Arena, Id, NARROW_WIDTH
from arena.paths import Paths
from arena.walk import Narrow, Prefix, Run, Walk, Wide
from bigint import BigUInt

from .args import Arguments, Signature, node_as_id, width_as_int32
from .capi import (
    Py_DecRef,
    Py_NewRef,
    Py_TPFLAGS_DEFAULT,
    Py_TPFLAGS_DISALLOW_INSTANTIATION,
    Py_tp_dealloc,
    Py_tp_free,
    Py_tp_iter,
    Py_tp_iternext,
    PyImport_ImportModule,
    PyList_New,
    PyList_SetItem,
    PyLong_FromLongLong,
    PyObject_CallFunctionObjArgs,
    PyObject_CallObject,
    PyObject_GetAttrString,
    PyTuple_New,
    PyTuple_SetItem,
    PyType_FromModuleAndSpec,
    PyType_GenericAlloc,
    PyType_GetSlot,
    PyUnicode_FromStringAndSize,
    TypeSlot,
    TypeSpec,
    address,
    ob_type,
    text,
)
from .errors import PyError
from .ints import Ints, c_function, int_from_big, int_from_u128
from .queries import checked_width
from .state import StatePointer, give_back, object_state, take_arena


# Batches start small, so taking a few items is quick, and double up to MAX_BATCH
# items or about MAX_BATCH_BITS bits.
comptime FIRST_BATCH = 16
comptime MAX_BATCH = 1024
comptime MAX_BATCH_BITS = 1 << 20


struct BatchSizes(TrivialRegisterPassable):
    var next: Int
    var most: Int

    def __init__(out self, width: Int32):
        self.most = max(min(MAX_BATCH, MAX_BATCH_BITS // max(Int(width), 1)), 1)
        self.next = min(FIRST_BATCH, self.most)

    def next_size(mut self) -> Int:
        var size = self.next
        self.next = min(size * 2, self.most)
        return size


@always_inline
def short(free: Int32, size: Int) -> Bool:
    """Whether a run of `2 ** free` members is shorter than `size`."""
    return free < 63 and 1 << Int(free) < size


struct Runs[V: Prefix](Movable):
    var walk: Walk[Self.V]
    var held: Optional[Run[Self.V]]
    """Didn't fit the last batch."""

    def __init__(out self, id: Id, width: Int32):
        self.walk = Walk[Self.V](id, width)
        self.held = None

    def next_batch(mut self, arena: Arena, size: Int) -> List[Run[Self.V]]:
        """A run of `size` members or more, or runs of up to `size` in all."""
        var runs = List[Run[Self.V]]()
        var listed = 0
        while True:
            var run: Run[Self.V]
            if self.held:
                run = self.held.take()
            else:
                var next = self.walk.next_run(arena)
                if not next:
                    break
                run = next.take()
            var long = not short(run.free, size)
            if long and not runs:
                runs.append(run^)
                break
            if long or listed + (1 << Int(run.free)) > size:
                self.held = run^
                break
            listed += 1 << Int(run.free)
            runs.append(run^)
        return runs^


comptime Walker = Variant[
    Runs[Narrow], Runs[Wide], Walk[Narrow], Walk[Wide], Paths
]


def members_walker(id: Id, width: Int32) -> Walker:
    if width <= NARROW_WIDTH:
        return Walker(Runs[Narrow](id, width))
    return Walker(Runs[Wide](id, width))


def blocks_walker(id: Id, width: Int32) -> Walker:
    if width <= NARROW_WIDTH:
        return Walker(Walk[Narrow](id, width))
    return Walker(Walk[Wide](id, width))


def paths_walker(id: Id, width: Int32) -> Walker:
    return Walker(Paths(id, width))


struct BatchesObject:
    """A batch iterator's layout: the object header, and the walk."""

    var ob_refcnt: Int
    var ob_type: PyObjectPtr
    var root: PyObjectPtr
    """Keeps the walk's nodes from being collected."""
    var walker: Walker
    var sizes: BatchSizes


@always_inline
def batches_of(o: PyObjectPtr) -> ref[MutUntrackedOrigin] BatchesObject:
    return o.bitcast[BatchesObject]().unsafe_value()[]


@always_inline
def int_of[V: Prefix](ints: Ints, value: V) -> PyObjectPtr:
    comptime if V == Narrow:
        return int_from_u128(ints, rebind[Narrow](value).value)
    else:
        return int_from_big(ints, rebind[Wide](value).value)


def pair(first: PyObjectPtr, second: PyObjectPtr) -> PyObjectPtr:
    """`(first, second)`, stealing both. Null if either is (after an error)."""
    var made = PyTuple_New(2) if first and second else PyObjectPtr()
    if not made:
        Py_DecRef(first)
        Py_DecRef(second)
        return made
    _ = PyTuple_SetItem(made, 0, first)
    _ = PyTuple_SetItem(made, 1, second)
    return made


def run_range[
    V: Prefix
](state: StatePointer, run: Run[V]) raises PyError -> PyObjectPtr:
    """`range(first, first + 2 ** free)`."""
    var length = BigUInt(1) << Int(run.free)
    var stop: BigUInt
    comptime if V == Narrow:
        stop = BigUInt(rebind[Narrow](run.first).value) + length
    else:
        stop = rebind[Wide](run.first).value + length
    var bounds = pair(
        int_of(state[].ints, run.first), int_from_big(state[].ints, stop)
    )
    if not bounds:
        raise PyError()
    var made = PyObject_CallObject(state[].range_type, bounds)
    Py_DecRef(bounds)
    if not made:
        raise PyError()
    return made


def next_members[
    V: Prefix
](
    state: StatePointer, mut runs: Runs[V], size: Int
) raises PyError -> PyObjectPtr:
    var arena = take_arena(state)
    var batch = runs.next_batch(arena[], size)
    give_back(state, arena)
    if not batch:
        return PyObjectPtr()
    if not short(batch[0].free, size):
        return run_range(state, batch[0])
    var total = 0
    for run in batch:
        total += 1 << Int(run.free)
    var list = PyList_New(total)
    if not list:
        raise PyError()
    var index = 0
    for run in batch:
        for n in range(1 << Int(run.free)):
            var member = int_of(state[].ints, run.first.plus(UInt64(n)))
            if not member:
                Py_DecRef(list)
                raise PyError()
            _ = PyList_SetItem(list, index, member)
            index += 1
    return list


def next_blocks[
    V: Prefix
](
    state: StatePointer, mut walk: Walk[V], size: Int
) raises PyError -> PyObjectPtr:
    var arena = take_arena(state)
    var runs = List[Run[V]]()
    while len(runs) < size:
        var run = walk.next_run(arena[])
        if not run:
            break
        runs.append(run.take())
    give_back(state, arena)
    if not runs:
        return PyObjectPtr()
    var list = PyList_New(len(runs))
    if not list:
        raise PyError()
    for index in range(len(runs)):
        ref run = runs[index]
        var block = pair(
            int_of(state[].ints, run.first),
            PyLong_FromLongLong(Int64(run.free)),
        )
        if not block:
            Py_DecRef(list)
            raise PyError()
        _ = PyList_SetItem(list, index, block)
    return list


def string_of(text: List[UInt8], start: Int, length: Int) -> PyObjectPtr:
    return PyUnicode_FromStringAndSize(
        StringSpan(unsafe_from_utf8=Span(text)[start : start + length])
    )


def next_paths(
    state: StatePointer, mut paths: Paths, size: Int
) raises PyError -> PyObjectPtr:
    var arena = take_arena(state)
    # Every path is `width` long, so they share one buffer.
    var text = List[UInt8]()
    var count = 0
    while count < size and paths.next_path(arena[]):
        text.extend(Span(paths.bits))
        count += 1
    give_back(state, arena)
    if count == 0:
        return PyObjectPtr()
    var list = PyList_New(count)
    if not list:
        raise PyError()
    var width = Int(paths.width)
    for index in range(count):
        var path = string_of(text, index * width, width)
        if not path:
            Py_DecRef(list)
            raise PyError()
        _ = PyList_SetItem(list, index, path)
    return list


def batches_next(o: PyObjectPtr) abi("C") -> PyObjectPtr:
    """The next batch, or null at the end (with no error set)."""
    ref batches = batches_of(o)
    var state = object_state(o)
    var size = batches.sizes.next_size()
    ref walker = batches.walker
    try:
        if walker.isa[Runs[Narrow]]():
            return next_members(state, walker[Runs[Narrow]], size)
        if walker.isa[Runs[Wide]]():
            return next_members(state, walker[Runs[Wide]], size)
        if walker.isa[Walk[Narrow]]():
            return next_blocks(state, walker[Walk[Narrow]], size)
        if walker.isa[Walk[Wide]]():
            return next_blocks(state, walker[Walk[Wide]], size)
        return next_paths(state, walker[Paths], size)
    except:
        return PyObjectPtr()


def batches_iter(o: PyObjectPtr) abi("C") -> PyObjectPtr:
    return Py_NewRef(o)


def batches_dealloc(o: PyObjectPtr) abi("C"):
    var type = ob_type(o)
    ref batches = batches_of(o)
    Pointer(to=batches.walker).unsafe_deinit_pointee()
    Py_DecRef(batches.root)
    var free = c_function[def(PyObjectPtr) thin abi("C") -> None](
        PyType_GetSlot(type, Py_tp_free)
    )
    free(o)
    Py_DecRef(type)


def from_iterable(state: StatePointer) raises PyError -> PyObjectPtr:
    """`itertools.chain.from_iterable` (borrowed)."""
    if state[].from_iterable:
        return state[].from_iterable
    var itertools = PyImport_ImportModule("itertools".as_c_string_span())
    if not itertools:
        raise PyError()
    var chain = PyObject_GetAttrString(itertools, "chain".as_c_string_span())
    Py_DecRef(itertools)
    if not chain:
        raise PyError()
    var method = PyObject_GetAttrString(
        chain, "from_iterable".as_c_string_span()
    )
    Py_DecRef(chain)
    if not method:
        raise PyError()
    state[].from_iterable = method
    return method


def walked[
    walker: def(Id, Int32) thin -> Walker
](state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    """The batches chained into one iterator."""
    var root = args[0]
    var id = node_as_id(state, root)
    var width = checked_width(state, id, width_as_int32(args[1]))
    var chain = from_iterable(state)
    var batches = PyType_GenericAlloc(state[].batches_type, 0)
    if not batches:
        raise PyError()
    ref made = batches_of(batches)
    made.root = Py_NewRef(root)
    Pointer(to=made.walker).unsafe_write(walker(id, width))
    made.sizes = BatchSizes(width)
    var chained = PyObject_CallFunctionObjArgs(chain, batches)
    Py_DecRef(batches)
    if not chained:
        raise PyError()
    return chained


comptime MEMBERS = Signature("members", positional=["node"], keywords=["width"])


def members(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    return walked[members_walker](state, args)


comptime BLOCKS = Signature(
    "blocks",
    positional=["node"],
    keywords=["width"],
    doc=(
        "Members as maximal aligned blocks (like CIDR blocks). `(first, free)`"
        " is the\n`2 ** free` integers from `first` on."
    ),
)


def blocks(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    return walked[blocks_walker](state, args)


comptime PATHS = Signature(
    "paths",
    positional=["node"],
    keywords=["width"],
    doc=(
        "Paths to `BDD.ACCEPT` as `width` characters of `0`, `1` or `?`, most"
        " significant\nfirst. Each member matches exactly one."
    ),
)


def paths(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    return walked[paths_walker](state, args)


def batches_type(module: PyObjectPtr) raises PyError -> PyObjectPtr:
    """The iterator type `members`, `blocks` and `paths` return, made at init. Its
    specs live as long as the process, as PyType_FromSpec needs."""
    var slots = unsafe_alloc[TypeSlot](4)
    slots[unsafe_offset=0] = TypeSlot(Py_tp_dealloc, address(batches_dealloc))
    slots[unsafe_offset=1] = TypeSlot(Py_tp_iter, address(batches_iter))
    slots[unsafe_offset=2] = TypeSlot(Py_tp_iternext, address(batches_next))
    slots[unsafe_offset=3] = TypeSlot(0, 0)
    var spec = unsafe_alloc[TypeSpec](1)
    spec[] = TypeSpec(
        text("bitpattern.bdd.Batches"),
        c_int(size_of[BatchesObject]()),
        0,
        # Else `Batches()` would make an iterator with no walk.
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_DISALLOW_INSTANTIATION,
        Int(slots),
    )
    var type = PyType_FromModuleAndSpec(module, Int(spec), PyObjectPtr())
    if not type:
        raise PyError()
    return type
