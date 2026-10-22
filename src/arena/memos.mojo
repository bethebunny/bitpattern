"""Operations' memoised results."""

from .table import IdTable, Table


struct Memos(Movable, Sized):
    """Emptying the tables never changes an answer. Symmetric operations key by
    `sorted_pair`."""

    var intersection: Table
    var union: Table
    var symmetric_difference: Table
    var complement: IdTable
    var subset: Table
    var disjoint: Table

    def __init__(out self):
        self.intersection = Table()
        self.union = Table()
        self.symmetric_difference = Table()
        self.complement = IdTable()
        self.subset = Table()
        self.disjoint = Table()

    def __len__(self) -> Int:
        var answers = len(self.subset) + len(self.disjoint)
        var made = len(self.intersection) + len(self.union)
        return (
            made
            + len(self.symmetric_difference)
            + len(self.complement)
            + answers
        )

    def capacity(self) -> Int:
        var answers = self.subset.capacity() + self.disjoint.capacity()
        var made = self.intersection.capacity() + self.union.capacity()
        return (
            made
            + self.symmetric_difference.capacity()
            + self.complement.capacity()
            + answers
        )

    def clear(mut self):
        """Each table keeps room for twice what it held, so similar walks needn't grow
        it."""
        self.intersection.clear()
        self.union.clear()
        self.symmetric_difference.clear()
        self.complement.clear()
        self.subset.clear()
        self.disjoint.clear()

    def shrink_to_fit(mut self):
        self.intersection.shrink_to_fit()
        self.union.shrink_to_fit()
        self.symmetric_difference.shrink_to_fit()
        self.complement.shrink_to_fit()
        self.subset.shrink_to_fit()
        self.disjoint.shrink_to_fit()
