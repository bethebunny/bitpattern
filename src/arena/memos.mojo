"""Operations' memoised results."""

from .table import IdTable, Table


struct Memos(Defaultable, Movable, Sized):
    """Emptying the tables never changes an answer."""

    var intersection: Table
    var union: Table
    var symmetric_difference: Table
    var complement: IdTable
    var subset: Table
    var disjoint: Table

    def __len__(self) -> Int:
        return (
            len(self.intersection)
            + len(self.union)
            + len(self.symmetric_difference)
            + len(self.complement)
            + len(self.subset)
            + len(self.disjoint)
        )

    def capacity(self) -> Int:
        return (
            self.intersection.capacity()
            + self.union.capacity()
            + self.symmetric_difference.capacity()
            + self.complement.capacity()
            + self.subset.capacity()
            + self.disjoint.capacity()
        )

    def clear(mut self):
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
