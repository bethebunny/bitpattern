"""Collecting nodes, and keeping one object per node as objects come and go."""

import _thread
import functools
import gc
import itertools
import operator
import os
import pickle
import random
import signal
import socket
import subprocess
import sys
import sysconfig
import time
import weakref

import pytest

from bitpattern import bdd
from bitpattern.bdd import BDD
from bitpattern.intset import IntSet


@pytest.fixture(autouse=True)
def only_live_nodes():
    """Starts each test from an arena that holds only live nodes."""
    gc.collect()
    bdd.collect()


def test_collect_frees_the_nodes_nothing_reaches():
    before = bdd.stats()["nodes"]
    temporary = IntSet(range(0, 1 << 16, 3), width=16)
    assert bdd.stats()["nodes"] > before
    del temporary
    assert bdd.collect() > 0
    assert bdd.stats()["nodes"] == before


def test_collect_keeps_the_nodes_live_objects_reach():
    group = IntSet(range(0, 1 << 16, 7), width=16)
    members = list(group)
    bdd.collect()
    assert list(group) == members
    assert IntSet(members, width=16).bdd is group.bdd


def test_the_arena_shrinks_back_once_a_big_set_is_gone():
    before = bdd.stats()
    rng = random.Random(0)
    big = IntSet([rng.getrandbits(64) for _ in range(20_000)], width=64)
    assert bdd.stats()["nodes"] > before["nodes"] + 100_000
    del big
    gc.collect()
    bdd.collect()
    after = bdd.stats()
    assert after["nodes"] == before["nodes"]
    assert after["free"] <= before["free"]
    assert after["room"] < before["nodes"] + before["free"] + 100_000
    assert after["memo_room"] == 0


def test_a_collection_on_its_own_keeps_room_for_the_nodes_it_freed():
    """So a workload that builds and drops big sets doesn't regrow the arena each time,
    while bdd.collect() gives the room back."""
    bdd.collect()
    rng = random.Random(0)
    IntSet([rng.getrandbits(64) for _ in range(30_000)], width=64)  # dropped at once
    made = bdd.stats()["nodes"]
    IntSet([1])  # collects first, since over 2**20 nodes were made
    stats = bdd.stats()
    assert stats["nodes"] < made // 2
    assert stats["room"] >= made
    bdd.collect()
    assert bdd.stats()["room"] < made // 2


def test_only_calls_that_make_nodes_collect():
    """A query leaves a collection that's due to the next call that makes a node, or to
    bdd.collect()."""
    node = IntSet([1, 5], width=8).bdd
    queries = [
        functools.partial(bdd.count, node, width=8),
        functools.partial(bdd.nth, node, 0, width=8),
        functools.partial(bdd.rank, node, 3, width=8),
        functools.partial(bdd.accepts, node, 5),
        functools.partial(bdd.subset, node, node),
        functools.partial(bdd.disjoint, node, node),
        functools.partial(bdd.node_count, node),
        functools.partial(bdd.path_count, node),
        functools.partial(repr, node),
        functools.partial(bdd.members, node, width=8),
        functools.partial(bdd.tensor, node, width=8, code=1, bits=8, version=None),
    ]
    rng = random.Random(0)
    IntSet([rng.getrandbits(64) for _ in range(30_000)], width=64)  # dropped at once
    made = bdd.stats()["nodes"]
    for query in queries:
        query()
    assert bdd.stats()["nodes"] == made
    assert bdd.collect() > 0


def test_reused_slots_dont_disturb_live_nodes():
    keep = IntSet([1, 5, 9], width=8)
    for width in (10, 11, 12):
        IntSet(range(0, 1 << width, 3), width=width)
        bdd.collect()
    assert list(keep) == [1, 5, 9]
    assert IntSet([9, 5, 1], width=8).bdd is keep.bdd


def test_objects_stay_canonical_across_collections():
    node = BDD(7, BDD.ACCEPT, BDD.REJECT)
    bdd.collect()
    assert BDD(7, BDD.ACCEPT, BDD.REJECT) is node


def test_collection_runs_as_the_arena_grows():
    """Without collecting, the arena would hold every set's nodes."""
    total = 0
    for seed in range(12):
        rng = random.Random(seed)
        group = IntSet([rng.getrandbits(64) for _ in range(10_000)], width=64)
        total += bdd.node_count(group.bdd)
    stats = bdd.stats()
    assert stats["nodes"] + stats["free"] < total / 2


def test_memo_tables_stay_bounded_when_no_nodes_are_made():
    """`&` of disjoint sets makes no nodes, but memoises the pairs of nodes it visits."""
    rng = random.Random(0)
    groups = [
        IntSet([rng.getrandbits(64) for _ in range(60)], width=64) for _ in range(200)
    ]
    nodes = bdd.stats()["nodes"]
    for a in groups:
        for b in groups:
            assert not a & b or a is b
    stats = bdd.stats()
    assert stats["nodes"] == nodes
    assert stats["memo"] <= max(1 << 20, nodes) + 10_000


def test_subset_and_disjoint_keep_their_answers_bounded():
    """They make no nodes, so they never make a collection due to empty the tables. Any
    two of these values share their top 16 bits, so each walk is long enough to keep its
    answer, and there are over 2**16 pairs of them."""
    values = random.Random(0).sample(range(1 << 16), 363)
    nodes = [bdd.interval(value, value + 1, width=32) for value in values]
    for compare in (bdd.subset, bdd.disjoint):
        memo = bdd.stats()["memo"]
        compare(nodes[0], nodes[1])
        assert bdd.stats()["memo"] == memo + 1
        sum(itertools.starmap(compare, itertools.combinations(nodes, 2)))
        assert bdd.stats()["memo"] <= memo + (1 << 16)


def test_subset_and_disjoint_remember_their_answers():
    """Asked again, they don't walk every pair of nodes again. An answer found in a few
    steps, eg. at the top bit for low and high here, isn't worth keeping."""
    rng = random.Random(0)
    a = IntSet([rng.getrandbits(32) for _ in range(1000)], width=32)
    b = a | IntSet([rng.getrandbits(32) for _ in range(1000)], width=32)
    c = IntSet([rng.getrandbits(32) for _ in range(1000)], width=32)
    low, high = IntSet([0], width=32), IntSet([1 << 31], width=32)
    before = bdd.stats()["memo"]
    assert a <= b and a.isdisjoint(c) and not b <= a and not low <= high
    memo = bdd.stats()["memo"]
    assert memo == before + 3
    assert a <= b and a.isdisjoint(c) and not b <= a and not low <= high
    assert bdd.stats()["memo"] == memo


def test_and_answers_subset_and_disjoint():
    rng = random.Random(0)
    a = IntSet([rng.getrandbits(32) for _ in range(1000)], width=32)
    b = IntSet([rng.getrandbits(32) for _ in range(1000)], width=32)
    both = a & b
    memo = bdd.stats()["memo"]
    assert a.isdisjoint(b) is (not both)
    assert (a <= b) is (both == a)
    assert bdd.stats()["memo"] == memo


def test_a_branch_has_one_object_while_it_lives():
    parent = BDD(1005, BDD(1004, BDD.ACCEPT, BDD.REJECT), BDD.REJECT)
    branch = parent.left
    gc.collect()
    bdd.collect()
    assert parent.left is branch
    assert (branch.bit, branch.left, branch.right) == (1004, BDD.ACCEPT, BDD.REJECT)


def test_walking_down_a_long_chain_then_dropping_it():
    """Objects don't keep their branches' objects, so dropping the top of a walk frees
    one object, not a chain of them, each inside the last one's deallocation."""
    top = node = IntSet([0], width=100_000).bdd
    while not node.leaf:
        node = node.left
    del top, node
    assert IntSet([1], width=30_000) == IntSet([1])  # walks the diagram to compare


def test_a_weakref_callback_can_make_the_node_whose_object_is_dying():
    made = []
    node = BDD(900, BDD.ACCEPT, BDD.REJECT)
    probe = weakref.ref(node, lambda _: made.append(BDD(900, BDD.ACCEPT, BDD.REJECT)))
    del node
    gc.collect()
    assert probe() is None
    assert made[0] is BDD(900, BDD.ACCEPT, BDD.REJECT)


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="needs free-threaded Python"
)
def test_free_threaded_python_turns_the_gil_on_for_the_arena():
    if sys.version_info >= (3, 13):
        assert sys._is_gil_enabled()


@pytest.mark.skipif(sys.version_info < (3, 14), reason="needs _interpreters.create")
@pytest.mark.parametrize("config", ["legacy", "isolated"])
def test_another_interpreter_gets_an_arena_of_its_own(config):
    """The arena and its nodes' objects belong to the interpreter that imported them,
    even one with a GIL of its own."""
    code = (
        "import _interpreters, bitpattern.bdd as bdd\n"
        "node = bdd.BDD(5, bdd.BDD.ACCEPT, bdd.BDD.REJECT)\n"
        f"other = _interpreters.create({config!r})\n"
        # Each interpreter has its own sys.stdout, so flush in order.
        "made = 'import bitpattern.bdd as b; "
        "print(b.BDD(3, b.BDD.ACCEPT, b.BDD.REJECT).size, flush=True)'\n"
        "assert _interpreters.exec(other, made) is None\n"
        "print(bdd.stats()['nodes'])\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, check=True)
    assert run.stdout == b"8\n3\n"


def test_the_same_interpreter_can_import_the_diagrams_again():
    code = (
        "import sys, bitpattern.bdd\n"
        "del sys.modules['bitpattern.bdd']\n"
        "import bitpattern.bdd\n"
        "print(bitpattern.bdd.BDD.ACCEPT)\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, check=True)
    assert run.stdout == b"BDD.ACCEPT\n"


class Stopped(Exception):
    """Raised by a test's SIGINT handler, so a stray one can't stop pytest itself."""


def interrupted(walk, handler):
    """Runs `walk` with Ctrl-C just pressed, under `handler`, so the walk's first pause
    takes it. Python takes a Ctrl-C at its next bytecode, so `walk` is a call into C, eg.
    a partial, and the garbage collector is off, since its callbacks, eg. pytest's, are
    Python too."""
    previous = signal.signal(signal.SIGINT, handler)
    gc.disable()
    try:
        steps = [_thread.interrupt_main, walk]  # no bytecode runs between the two
        _, result = map(operator.methodcaller("__call__"), steps)
        return result
    finally:
        gc.enable()
        signal.signal(signal.SIGINT, previous)


def interrupted_negation(handler):
    """Negates a big diagram with Ctrl-C just pressed, under `handler`."""
    rng = random.Random(0)
    group = IntSet([rng.getrandbits(64) for _ in range(20_000)], width=64)
    bdd.collect()  # empties the memo tables, so ~ walks the whole diagram
    return group, interrupted(functools.partial(operator.invert, group.bdd), handler)


def test_ctrl_c_stops_a_long_walk_partway():
    """Python's handler runs with the arena unlocked, so it can use diagrams too."""
    seen = []

    def handler(signum, frame):
        seen.append(bdd.stats()["nodes"])
        raise Stopped

    with pytest.raises(Stopped):
        interrupted_negation(handler)
    rng = random.Random(0)
    group = IntSet([rng.getrandbits(64) for _ in range(20_000)], width=64)
    negated = ~group.bdd  # this time to the end
    assert bdd.stats()["nodes"] > seen[0]
    assert negated & group.bdd is BDD.REJECT


def test_ctrl_c_stops_unpickling_a_big_diagram_partway():
    rng = random.Random(0)
    pickled = pickle.dumps(IntSet([rng.getrandbits(64) for _ in range(20_000)]).bdd)
    bdd.collect()  # so unpickling makes every node again
    seen = []

    def handler(signum, frame):
        seen.append(bdd.stats()["nodes"])
        raise Stopped

    with pytest.raises(Stopped):
        interrupted(functools.partial(pickle.loads, pickled), handler)
    assert pickle.loads(pickled) is not None
    assert 2 < seen[0] < bdd.stats()["nodes"]


def test_a_walk_goes_on_when_the_ctrl_c_handler_doesnt_raise():
    seen = []
    group, negated = interrupted_negation(lambda signum, frame: seen.append(signum))
    assert seen == [signal.SIGINT]
    assert ~negated is group.bdd
    assert negated & group.bdd is BDD.REJECT


def test_the_ctrl_c_handlers_exception_comes_out_of_the_walk():
    """Fetching an exception from Python can run Python code, eg. its metaclass's
    __subclasscheck__ on 3.11, which mustn't run while the walk has the arena."""

    class Checked(type):
        def __subclasscheck__(cls, subclass):
            bdd.stats()  # uses the arena
            return super().__subclasscheck__(subclass)

    class Raised(Stopped, metaclass=Checked):
        pass

    def handler(signum, frame):
        raise Raised

    with pytest.raises(Raised):
        interrupted_negation(handler)


def test_an_error_runs_no_python_while_its_call_has_the_arena():
    """On 3.11, an error set while another is handled makes its exception at once, which
    can start the garbage collector, whose callbacks are Python."""
    node = IntSet([1, 5], width=8).bdd
    tensor = functools.partial(bdd.tensor, code=1, bits=8, version=None)
    calls = [
        functools.partial(bdd.count, node, width=2),
        functools.partial(bdd.nth, node, 0, width=2),
        functools.partial(bdd.rank, node, 0, width=2),
        functools.partial(bdd.members, node, width=2),
        functools.partial(tensor, node, width=2),
        functools.partial(tensor, BDD.ACCEPT, width=99),  # too many to fit in memory
        functools.partial(BDD, 0, node, BDD.REJECT),  # misordered
        functools.partial(bdd.lshift, node, -3),  # moves bit 0 out
        functools.partial(bdd.lshift, node, 1 << 40),  # too far for any bit
        functools.partial(bdd._rebuilt, [(0, 0, 1), (0, 2, 1)]),  # misordered
    ]
    failed = []

    def use_the_arena(phase, info):
        try:
            bdd.stats()
        except RuntimeError:
            failed.append(phase)

    def call_handling_another(call):
        try:
            raise KeyError
        except KeyError:
            call()

    threshold = gc.get_threshold()
    gc.callbacks.append(use_the_arena)
    gc.set_threshold(1)  # collects at almost every allocation
    try:
        for call in calls:
            with pytest.raises((ValueError, MemoryError)):
                call_handling_another(call)
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(use_the_arena)
    assert failed == []


def test_collection_runs_again_after_a_call_raises():
    """Nothing collects while a call that makes nodes is underway, so an error has to
    end its call."""
    node = IntSet([1, 5], width=8).bdd
    with pytest.raises(ValueError):
        BDD(0, node, BDD.REJECT)
    with pytest.raises(ValueError):
        bdd.lshift(node, -3)
    with pytest.raises(ValueError):
        bdd.lshift(node, 1 << 40)
    with pytest.raises(ValueError):
        bdd._rebuilt([(0, 0, 1), (0, 2, 1)])
    IntSet(range(0, 1 << 16, 3), width=16)  # garbage for a collection to free
    assert bdd.collect() > 0


def test_a_base_exception_out_of_the_ctrl_c_handler_keeps_the_arena():
    """One that isn't an Exception (as PyO3's PanicException isn't) stops the walk like
    any other."""

    class Halted(BaseException):
        pass

    def handler(signum, frame):
        raise Halted

    rng = random.Random(0)
    group = IntSet([rng.getrandbits(64) for _ in range(20_000)], width=64)
    bdd.collect()  # empties the memo tables, so ~ walks the whole diagram
    nodes = bdd.stats()["nodes"]
    with pytest.raises(Halted):
        interrupted(functools.partial(operator.invert, group.bdd), handler)
    assert bdd.stats()["nodes"] >= nodes
    assert BDD.REJECT.leaf
    assert ~group.bdd & group.bdd is BDD.REJECT


def test_nothing_collects_while_a_ctrl_c_handler_runs_in_a_walk():
    """The walk can hold nodes nothing else reaches, eg. once the handler drops the
    object for the node it started from."""
    rng = random.Random(0)
    group = IntSet([rng.getrandbits(64) for _ in range(20_000)], width=64)
    IntSet(range(0, 1 << 16, 3), width=16)  # garbage for a collection to free
    freed = []
    walk = functools.partial(bdd.node_count, group.bdd)
    nodes = interrupted(walk, lambda signum, frame: freed.append(bdd.collect()))
    assert freed == [0]
    assert nodes == bdd.node_count(group.bdd)


def test_ctrl_c_stops_a_dlpack_export_partway():
    """A set can have billions of members, so an export pauses as it writes them."""
    group = IntSet(range(0, 1 << 16, 2), width=16)  # a run, and a step, per member
    IntSet(range(1, 1 << 16, 3), width=16)  # garbage for a collection to free
    freed = []

    def handler(signum, frame):
        freed.append(bdd.collect())
        raise Stopped

    export = functools.partial(
        bdd.tensor, group.bdd, width=16, code=1, bits=16, version=(1, 0)
    )
    with pytest.raises(Stopped):
        interrupted(export, handler)
    assert freed == [0]  # so the handler ran mid-walk, where nothing collects


def test_ctrl_c_stops_a_dlpack_export_partway_through_one_run():
    """An export writes a run a chunk at a time, and pauses between chunks."""
    group = IntSet.range(0, 1 << 26, width=26)  # a pause comes at least every 2 ** 26
    IntSet(range(1, 1 << 16, 3), width=16)  # garbage for a collection to free
    freed = []

    def handler(signum, frame):
        freed.append(bdd.collect())
        raise Stopped

    export = functools.partial(
        bdd.tensor, group.bdd, width=26, code=1, bits=32, version=(1, 0)
    )
    with pytest.raises(Stopped):
        interrupted(export, handler)
    assert freed == [0]


@pytest.mark.skipif(sys.platform == "win32", reason="needs kill")
def test_a_walk_goes_on_from_where_a_ctrl_c_paused_it():
    """However often the handler runs, the walk doesn't start over, so it finishes."""
    seen = []
    rng = random.Random(1)
    values = [rng.getrandbits(64) for _ in range(20_000)]
    pid = os.getpid()
    previous = signal.signal(
        signal.SIGINT, lambda signum, frame: seen.append(bdd.stats()["nodes"])
    )
    try:
        presser = subprocess.Popen(
            ["sh", "-c", f"for _ in $(seq 400); do kill -INT {pid}; sleep 0.005; done"]
        )
        try:
            bdd.exactly_these(values, width=64)  # a walk that keeps no memo
            made = bdd.stats()["nodes"]
        finally:
            presser.kill()
            presser.wait()
            time.sleep(0.05)  # for any Ctrl-C still on its way
    finally:
        signal.signal(signal.SIGINT, previous)
    during = [nodes for nodes in seen if nodes < made]
    assert len(during) > 2
    assert during == sorted(set(during))  # each pause found the walk further on


def test_a_ctrl_c_during_a_walk_writes_to_the_wakeup_fd_once():
    """An event loop takes each byte there for a signal."""
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.setblocking(False)
        writer.setblocking(False)
        wakeup = signal.set_wakeup_fd(writer.fileno())
        try:
            interrupted_negation(lambda signum, frame: None)
        finally:
            signal.set_wakeup_fd(wakeup)
        assert reader.recv(16) == bytes([signal.SIGINT])
