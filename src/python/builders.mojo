"""Functions that make diagrams."""

from std.bit import count_leading_zeros
from std.collections import Set
from std.ffi import c_int, c_uint
from std.python._cpython import PyObjectPtr

from arena import (
    ACCEPT,
    Arena,
    Id,
    Interrupted,
    MAX_WIDTH,
    Misordered,
    NARROW_WIDTH,
    REJECT,
)
from arena import build, reshape, rows
from arena.reshape import ShiftedOut
from arena.rows import Row
from bigint import BigUInt

from .args import (
    Arguments,
    Signature,
    bit_as_int32,
    clamped,
    int32_up_to,
    least_width_as_int32,
    node_as_id,
    not_an_instance,
    str_or_size,
    utf8,
    width_as_int32,
)
from .capi import (
    Py_BuildValue,
    Py_DecRef,
    Py_NewRef,
    Py_TPFLAGS_LIST_SUBCLASS,
    Py_TPFLAGS_TUPLE_SUBCLASS,
    Py_TPFLAGS_UNICODE_SUBCLASS,
    PyErr_Clear,
    PyErr_Occurred,
    PyImport_ImportModule,
    PyIter_Next,
    PyList_New,
    PyList_SetItem,
    PyLong_FromLongLong,
    PyObject_GetAttrString,
    PyObject_GetIter,
    PyObject_IsInstance,
    PyObject_Repr,
    PyTuple_GetItem,
    PyTuple_New,
    PyTuple_SetItem,
    PyTuple_Size,
    PyType_GetFlags,
    PyType_GetModule,
    PyUnicode_AsUTF8AndSize,
    PyUnicode_FromOrdinal,
    PyUnicode_InternFromString,
    ob_type,
)
from .errors import PyError, type_error, value_error
from .ints import Int64OrOverflow, Ints, Signed, indexed, u128_of
from .objects import Call, node_of, wrap, wrap_result
from .state import Lease, StatePointer, give_back, take_arena


def reduced(state: StatePointer, o: PyObjectPtr) raises PyError -> PyObjectPtr:
    """BDD.__reduce__: leaves pickle by name, and other nodes as flat rows, since
    nesting would recurse once per bit."""
    var id = node_of(o)
    if id.is_leaf():
        var name: PyObjectPtr
        if id == ACCEPT:
            name = PyUnicode_InternFromString("BDD.ACCEPT".as_c_string_span())
        else:
            name = PyUnicode_InternFromString("BDD.REJECT".as_c_string_span())
        if not name:
            raise PyError()
        return name
    var lease = Lease(state)
    var found: List[Row]
    try:
        with lease^ as arena:
            found = rows.rows(arena[], id)
    except:
        raise PyError()
    var written = PyList_New(len(found))
    if not written:
        raise PyError()
    for index in range(len(found)):
        var row = found[index]
        var item = Py_BuildValue(
            "(iII)".as_c_string_span(),
            c_int(row.bit),
            c_uint(row.lo),
            c_uint(row.hi),
        )
        # Steals `item`, even if it fails.
        if not item or PyList_SetItem(written, index, item) < 0:
            Py_DecRef(written)
            raise PyError()
    var module = PyType_GetModule(ob_type(o))
    var rebuild = PyObject_GetAttrString(module, "_rebuilt".as_c_string_span())
    if not rebuild:
        Py_DecRef(written)
        raise PyError()
    var reduction = Py_BuildValue("(N(N))".as_c_string_span(), rebuild, written)
    if not reduction:
        raise PyError()
    return reduction


comptime LSHIFT = Signature("lshift", positional=["node", "by"])


def lshift(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var id = node_as_id(state, args[0])
    var by = Int64OrOverflow.from_python(args[1])
    if not by.fits():
        # Too far for any decision node.
        if id.is_leaf():
            return Py_NewRef(args[0])
        var arena = take_arena(state)
        var top = arena[].decision(id).bit
        give_back(state, arena)
        # `str_or_size` can run Python, so the arena goes back first.
        raise shifted_out(str_or_size(args[1]), top)
    if by.value == 0 or id.is_leaf():
        return Py_NewRef(args[0])
    if by.value < Int64(Int32.MIN) or by.value > Int64(Int32.MAX):
        var arena = take_arena(state)
        var top = arena[].decision(id).bit
        give_back(state, arena)
        raise shifted_out(String(by.value), top)
    with Call(state):
        var lease = Lease(state)
        var moved: Id
        try:
            with lease^ as arena:
                moved = reshape.lshift(arena[], id, Int32(by.value))
        except e:
            if e.isa[ShiftedOut]():
                raise shifted_out(String(by.value), e[ShiftedOut].bit)
            raise PyError()  # the Ctrl-C handler's
        return wrap(state, moved)


def shifted_out(by: String, bit: Int32) -> PyError:
    return value_error(
        String("shifting by ", by, " moves bit ", bit, " out of range")
    )


comptime FREED_BETWEEN = Signature(
    "freed_between",
    positional=["node"],
    keywords=["low_bit", "high_bit"],
    doc=(
        "Accepts `i` when `node` accepts any `i'` with the same bits outside\n"
        "`[low_bit, high_bit)`."
    ),
)


def freed_between(
    state: StatePointer, args: Arguments
) raises PyError -> PyObjectPtr:
    var id = node_as_id(state, args[0])
    var low = clamped(args[1])
    var high = clamped(args[2])
    if low >= high:
        return Py_NewRef(args[0])

    def walk(
        mut arena: Arena,
    ) raises Interrupted {var id, var low, var high} -> Id:
        return reshape.freed_between(arena, id, low_bit=low, high_bit=high)

    return wrap_result(state, walk)


comptime NARROWED = Signature(
    "narrowed", positional=["node"], keywords=["width"]
)


def narrowed(
    state: StatePointer, args: Arguments
) raises PyError -> PyObjectPtr:
    var id = node_as_id(state, args[0])
    var width = width_as_int32(args[1])
    var arena = take_arena(state)
    var narrow = arena[].tests_only_below(id, width)
    give_back(state, arena)
    if narrow:
        return Py_NewRef(args[0])

    def walk(mut arena: Arena) raises Interrupted {var id, var width} -> Id:
        return reshape.narrowed(arena, id, width)

    return wrap_result(state, walk)


comptime DIAGRAM = Signature(
    "diagram",
    positional=["bits"],
    doc="`bits` is most significant first, with `?` for free bits.",
)


def diagram(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var o = args[0]
    if PyType_GetFlags(ob_type(o)) & Py_TPFLAGS_UNICODE_SUBCLASS == 0:
        raise not_an_instance(state, o, "str")
    var size = 0
    var data = PyUnicode_AsUTF8AndSize(o, size)
    if data == 0:
        raise PyError()
    # The str keeps its UTF-8 for as long as it lives, and the caller holds it.
    var bits = StringSlice(
        unsafe_from_utf8=Span[UInt8, ImmUntrackedOrigin](
            unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](
                unsafe_from_address=data
            ),
            length=size,
        )
    )
    var bytes = bits.as_bytes()
    for index in range(size):
        var byte = bytes[index]
        if (
            byte != UInt8(ord("0"))
            and byte != UInt8(ord("1"))
            and byte != UInt8(ord("?"))
        ):
            # Bits are ASCII, so the first byte that isn't one starts a character.
            var decoded = Codepoint.unsafe_decode_utf8_codepoint(bytes[index:])
            raise value_error(
                String("bits are 0, 1 or ?, not ", repr_of(decoded[0]))
            )
    if size > Int(MAX_WIDTH):
        raise value_error(String(size, " bits is too many"))

    def walk(mut arena: Arena) raises Interrupted {var bits} -> Id:
        return build.diagram(arena, bits)

    return wrap_result(state, walk)


def repr_of(symbol: Codepoint) -> String:
    """Python's `repr(chr(symbol))`, or "?" if it can't be made."""
    var char = PyUnicode_FromOrdinal(c_int(symbol.to_u32()))
    if not char:
        PyErr_Clear()
        return "?"
    var text = PyObject_Repr(char)
    Py_DecRef(char)
    if not text:
        PyErr_Clear()
        return "?"
    var written = utf8(text)
    Py_DecRef(text)
    return written^


comptime EXACTLY_THESE = Signature(
    "exactly_these",
    positional=["values"],
    keywords=["width"],
    doc=(
        "The set of `values`, and its width (at least `width`). Errors are"
        " IntSet's."
    ),
)


def exactly_these(
    state: StatePointer, args: Arguments
) raises PyError -> PyObjectPtr:
    var least_width = least_width_as_int32(args[1])
    var members = Members.read(state[].ints, args[0])
    var bits = members.bits()
    if bits > Int(MAX_WIDTH):
        raise value_error(
            String("a value of ", bits, " bits is wider than any set")
        )
    var width = max(least_width, Int32(bits))

    def walk(
        mut arena: Arena,
    ) raises Interrupted {imm members, var width} -> Id:
        if members.wide:
            return build.exactly_these[build.wide_bit](
                arena, members.wide.value(), width
            )
        return build.exactly_these[build.narrow_bit](
            arena, members.narrow, width
        )

    var node = wrap_result(state, walk)
    var pair = PyTuple_New(2)
    var width_int = PyLong_FromLongLong(Int64(width))
    if not pair or not width_int:
        Py_DecRef(node)
        Py_DecRef(pair)
        Py_DecRef(width_int)
        raise PyError()
    _ = PyTuple_SetItem(pair, 0, node)
    _ = PyTuple_SetItem(pair, 1, width_int)
    return pair


struct Distinct[T: KeyElement & Copyable & Comparable & Deinitable](Movable):
    """Values without repeats. The list is deduplicated each time it doubles and becomes
    a hash set once most values were repeats."""

    var values: List[Self.T]
    var sorted_length: Int
    """`values` is sorted and distinct up to here."""
    var pushed: Int
    """Counts repeats too."""
    var hashed: Set[Self.T]
    var hashing: Bool

    def __init__(out self):
        self.values = List[Self.T]()
        self.sorted_length = 0
        self.pushed = 0
        self.hashed = Set[Self.T]()
        self.hashing = False

    def __init__(out self, *, var sorted: List[Self.T]):
        """`sorted` is sorted and distinct."""
        self.sorted_length = len(sorted)
        self.pushed = len(sorted)
        self.values = sorted^
        self.hashed = Set[Self.T]()
        self.hashing = False

    def push(mut self, var value: Self.T):
        if self.hashing:
            self.hashed.add(value^)
            return
        self.values.append(value^)
        self.pushed += 1
        if len(self.values) >= 2 * max(self.sorted_length, 1024):
            deduplicate(self.values)
            self.sorted_length = len(self.values)
            if 2 * len(self.values) < self.pushed:
                for value in self.values:
                    self.hashed.add(value.copy())
                self.hashing = True
                self.values = List[Self.T]()

    def sorted(deinit self) -> List[Self.T]:
        var values = self.values^
        for value in self.hashed:
            values.append(value.copy())
        deduplicate(values)
        return values^


def deduplicate[T: Copyable & Comparable & Deinitable](mut values: List[T]):
    comptime if T == UInt128:
        radix_sort(rebind[List[UInt128]](values))
    else:
        sort(Span(values))
    var kept = 0
    for index in range(len(values)):
        if kept == 0 or values[kept - 1] != values[index]:
            values[kept] = values[index].copy()
            kept += 1
    values.shrink(kept)


def radix_sort(mut values: List[UInt128]):
    """Least significant byte first."""
    var n = len(values)
    # Counting costs more than comparing a few values.
    if n < 64:
        sort(Span(values))
        return
    # How many values have each byte at each of the 16 places.
    var counts = List[Int](length=16 * 256, fill=0)
    for value in values:
        comptime for place in range(16):
            counts.unsafe_get(256 * place + byte_at(value, place)) += 1
    var moved = List[UInt128](length=n, fill=0)
    for place in range(16):
        var start = 256 * place
        # Every value has the same byte here, so the pass would move nothing.
        if counts.unsafe_get(start + byte_at(values.unsafe_get(0), place)) == n:
            continue
        var total = 0
        for index in range(start, start + 256):
            var count = counts.unsafe_get(index)
            counts.unsafe_set(index, total)
            total += count
        for value in values:
            ref at = counts.unsafe_get(start + byte_at(value, place))
            moved.unsafe_set(at, value)
            at += 1
        swap(values, moved)


@always_inline
def byte_at(value: UInt128, place: Int) -> Int:
    return Int((value >> UInt128(8 * place)) & 0xFF)


struct Members(Movable):
    """Sorted and distinct. In UInt128s, which are smaller and sort faster, unless a
    value doesn't fit one."""

    var narrow: List[UInt128]
    var wide: Optional[List[BigUInt]]

    @staticmethod
    def read(ints: Ints, values: PyObjectPtr) raises PyError -> Members:
        var iterator = PyObject_GetIter(values)
        if not iterator:
            raise PyError()
        var narrow = Distinct[UInt128]()
        var wide: Optional[Distinct[BigUInt]] = None
        while True:
            var item = PyIter_Next(iterator)
            if not item:
                break
            var int: PyObjectPtr
            try:
                # Runs __index__ once, so a second read can't hide the first's error.
                int = indexed(ints, item)
            except e:
                Py_DecRef(item)
                Py_DecRef(iterator)
                raise e
            Py_DecRef(item)
            var small = u128_of(ints, int) if not wide else None
            if small:
                narrow.push(small.value())
                Py_DecRef(int)
                continue
            var signed = Signed.from_int(ints, int)
            if signed.negative:
                var error = value_error(
                    String(
                        "IntSets can't hold negative numbers, got ",
                        str_or_size(int),
                    )
                )
                Py_DecRef(int)
                Py_DecRef(iterator)
                raise error
            Py_DecRef(int)
            if not wide:
                var promoted = List[BigUInt]()
                for value in narrow^.sorted():
                    promoted.append(BigUInt(value))
                narrow = Distinct[UInt128]()
                wide = Distinct[BigUInt](sorted=promoted^)
            wide.value().push(signed.magnitude.copy())
        Py_DecRef(iterator)
        if PyErr_Occurred():
            raise PyError()
        if wide:
            return Members(List[UInt128](), wide.take().sorted())
        return Members(narrow^.sorted(), None)

    def __init__(
        out self, var narrow: List[UInt128], var wide: Optional[List[BigUInt]]
    ):
        self.narrow = narrow^
        self.wide = wide^

    def bits(self) -> Int:
        if self.wide:
            ref values = self.wide.value()
            return values[len(values) - 1].bits() if values else 0
        if not self.narrow:
            return 0
        return 128 - Int(count_leading_zeros(self.narrow[len(self.narrow) - 1]))


comptime INTERVAL = Signature(
    "interval", positional=["start", "stop"], keywords=["width"]
)


def interval(
    state: StatePointer, args: Arguments
) raises PyError -> PyObjectPtr:
    var width = width_as_int32(args[2])
    var start = Int64OrOverflow.from_python(args[0])
    var stop = Int64OrOverflow.from_python(args[1])
    # Bounds that fit an Int64 make members that fit a UInt128.
    if start.fits() and stop.fits():
        if stop.value <= 0:
            return wrap(state, REJECT)
        var top = UInt128(0)
        if width > 0:
            top = UInt128.MAX >> UInt128(128 - min(width, 128))
        var first = UInt128(max(start.value, 0))
        var last = min(UInt128(stop.value - 1), top)
        if first > last:
            return wrap(state, REJECT)
        return narrow_interval(state, first, last, width)
    var low = Signed.from_python(state[].ints, args[0])
    var high = Signed.from_python(state[].ints, args[1])
    if high.negative or not high.magnitude:
        return wrap(state, REJECT)
    var last = high.magnitude - BigUInt(1)
    if last.bits() > Int(width):
        last = BigUInt.mask(Int(width))
    var first = BigUInt() if low.negative else low.magnitude.copy()
    if first > last:
        return wrap(state, REJECT)
    if width <= NARROW_WIDTH:
        return narrow_interval(
            state,
            first.uint128_or_none().value(),
            last.uint128_or_none().value(),
            width,
        )

    def walk(
        mut arena: Arena,
    ) raises Interrupted {imm first, imm last, var width} -> Id:
        return build.interval[build.wide_bit](arena, first, last, width)

    return wrap_result(state, walk)


def narrow_interval(
    state: StatePointer, first: UInt128, last: UInt128, width: Int32
) raises PyError -> PyObjectPtr:
    def walk(
        mut arena: Arena,
    ) raises Interrupted {var first, var last, var width} -> Id:
        return build.interval[build.narrow_bit](arena, first, last, width)

    return wrap_result(state, walk)


comptime REBUILT = Signature("_rebuilt", positional=["rows"])


def rebuilt(state: StatePointer, args: Arguments) raises PyError -> PyObjectPtr:
    var given = read_rows(state, args[0])
    if not given or not formed(given):
        raise value_error("rows don't make a diagram")
    with Call(state):
        var lease = Lease(state)
        var built: Id
        try:
            with lease^ as arena:
                built = rows.rebuilt(arena[], given)
        except e:
            if e.isa[Misordered]():
                raise value_error(String(e[Misordered]))
            raise PyError()  # the Ctrl-C handler's
        return wrap(state, built)


def formed(rows: List[Row]) -> Bool:
    """Whether every branch names a leaf or an earlier row."""
    for index in range(len(rows)):
        # Row `index` makes node `2 + index`.
        if Int(rows[index].lo) >= 2 + index or Int(rows[index].hi) >= 2 + index:
            return False
    return True


def read_rows(state: StatePointer, o: PyObjectPtr) raises PyError -> List[Row]:
    """A sequence of `(bit, lo, hi)` tuples."""
    var flags = PyType_GetFlags(ob_type(o))
    if flags & Py_TPFLAGS_UNICODE_SUBCLASS:
        raise type_error("rows can't be a str")
    var listed = flags & (Py_TPFLAGS_LIST_SUBCLASS | Py_TPFLAGS_TUPLE_SUBCLASS)
    if not listed and not is_sequence(o):
        raise not_an_instance(state, o, "Sequence")
    var iterator = PyObject_GetIter(o)
    if not iterator:
        raise PyError()
    var rows = List[Row]()
    while True:
        var item = PyIter_Next(iterator)
        if not item:
            break
        try:
            rows.append(read_row(state, item))
        except e:
            Py_DecRef(item)
            Py_DecRef(iterator)
            raise e
        Py_DecRef(item)
    Py_DecRef(iterator)
    if PyErr_Occurred():
        raise PyError()
    return rows^


def is_sequence(o: PyObjectPtr) raises PyError -> Bool:
    """`isinstance(o, collections.abc.Sequence)`."""
    var abc = PyImport_ImportModule("collections.abc".as_c_string_span())
    if not abc:
        raise PyError()
    var sequence = PyObject_GetAttrString(abc, "Sequence".as_c_string_span())
    Py_DecRef(abc)
    if not sequence:
        raise PyError()
    var found = PyObject_IsInstance(o, sequence)
    Py_DecRef(sequence)
    if found < 0:
        raise PyError()
    return found == 1


def read_row(state: StatePointer, o: PyObjectPtr) raises PyError -> Row:
    """A branch out of range reads as UInt32.MAX, which `formed` rejects."""
    if PyType_GetFlags(ob_type(o)) & Py_TPFLAGS_TUPLE_SUBCLASS == 0:
        raise not_an_instance(state, o, "tuple")
    if PyTuple_Size(o) != 3:
        raise value_error(
            String(
                "expected tuple of length 3, but got tuple of length ",
                PyTuple_Size(o),
            )
        )
    var bit = bit_as_int32(PyTuple_GetItem(o, 0))
    var lo = int32_up_to(PyTuple_GetItem(o, 1), Int32.MAX)
    var hi = int32_up_to(PyTuple_GetItem(o, 2), Int32.MAX)
    return Row(
        bit,
        UInt32(lo.value()) if lo is not None else UInt32.MAX,
        UInt32(hi.value()) if hi is not None else UInt32.MAX,
    )
