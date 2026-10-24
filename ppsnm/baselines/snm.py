"""Sorted Neighbourhood Method baseline."""

from .base import _BaselineBase


class SortedNeighbourhoodMethod(_BaselineBase):
    """Traditional SNM: slide the window over the sorted records (record-first order)."""

    def _traverse(self, order, index_list, cols):
        N = len(order)
        for i in range(N):
            for dist in range(1, min(self.W - 1, N - 1 - i) + 1):
                self._compare(order, index_list, i, i + dist, cols)
