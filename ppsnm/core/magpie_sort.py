"""Progressive sorting utility."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _factorized_key_codes(data, key):
    """Return sortable codes with missing values ordered after valid values."""
    codes, _ = pd.factorize(data[key], sort=True)
    codes = codes.astype(np.int64)
    codes[codes < 0] = codes.max() + 1
    return codes


class MagpieSort:
    """Progressive sort used by the paper's sorted-neighbourhood variants.

    The algorithm emits a correctly sorted prefix of the data after each pass. This allows
    downstream algorithms to start comparing records before a complete order is known.

    Parameters
    ----------
    D : pandas.DataFrame
        Input table containing the records to sort.
    key : str
        Column name used to establish the sort order.
    pSize : int
        Maximum number of records kept in the working buffer per pass.
    W : int, default=0
        Window size. When ``W > 1``, an overlap between consecutive passes is preserved.

    Notes
    -----
    Like selection sort, the data is scanned repeatedly. Each pass collects the ``pSize``
    smallest remaining records (the magpie buffer) in sorted order. The first
    ``step = pSize - W + 1`` records are final and emitted; the remaining ``W - 1`` records
    are re-collected by the next pass, which gives the partitions the overlap needed by PSNM.
    The sort keys are factorized once to get a stable total order identical to a sorted
    column with ``NaN`` values placed last.
    """

    def __init__(self, D, key, pSize, W=0):
        if pSize < 1:
            raise ValueError("pSize must be at least 1")
        if W > 0 and pSize < W:
            raise ValueError(f"pSize must be at least W={W}")
        self.D = D
        self.key = key
        self.pSize = pSize
        self.W = W
        self.index_list = D.index.to_numpy()
        self.step = min(pSize, max(1, pSize - W + 1))

    def _composite_keys(self):
        N = len(self.D)
        codes = _factorized_key_codes(self.D, self.key)
        return codes * N + np.arange(N, dtype=np.int64)

    def passes(self) -> Iterable[tuple[int, np.ndarray]]:
        """Yield sorted partitions emitted by the progressive sort.

        Yields
        ------
        offset : int
            Starting offset of the emitted partition in the final order.
        part : numpy.ndarray
            Positional row ids of the sort ranks in this emitted partition.
        """
        N = len(self.D)
        comp = self._composite_keys()
        emitted = np.zeros(N, dtype=bool)
        offset = 0
        while offset < N:
            cand = np.flatnonzero(~emitted)  # records still to be placed
            c = comp[cand]
            k = min(self.pSize, len(cand))
            top = np.argpartition(c, k - 1)[:k] if k < len(cand) else np.arange(len(cand))
            top = top[np.argsort(c[top], kind="stable")]
            part = cand[top]

            n_emit = len(part) if len(cand) <= self.pSize else self.step
            emitted[part[:n_emit]] = True
            yield offset, part
            offset += n_emit

    def fit(self, on_partition=None):
        """Fit the progressive sort and return the final ordering state.

        After each pass, ``on_partition(p, offset, order, plen, known)`` is called. The final
        order array is only partially populated while the algorithm progresses; the prefix of the
        array up to ``known`` is already final and can be compared immediately.

        Returns
        -------
        tuple
            ``(order, index_list, partitions, pNum)`` where ``partitions[p]`` is the emitted
            partition view for pass ``p``.
        """
        N = len(self.D)
        order = np.full(N, -1, dtype=np.int64)
        partitions = {}
        for p, (offset, part) in enumerate(self.passes()):
            known = offset + len(part)
            order[offset:known] = part
            partitions[p] = order[offset:known]
            if on_partition is not None:
                on_partition(p, offset, order, len(part), known)
        return order, self.index_list, partitions, len(partitions)

    def run(self, on_partition=None):
        """Backward-compatible alias for ``fit``."""
        return self.fit(on_partition=on_partition)
