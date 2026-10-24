"""Progressive blocking duplicate-detection algorithm."""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from ..core.evaluation import ProgressiveEvaluation
from ..core.incremental_transitive_closure import IncrementalTransitiveClosure
from ..core.magpie_sort import MagpieSort


class ProgressiveBlocking(ProgressiveEvaluation):
    """Progressive Blocking (PB) with attribute concurrency.

    Parameters
    ----------
    D : pandas.DataFrame
        Input records.
    K : str or list of str
        Key column name, or multiple key columns for attribute concurrency.
    R : int
        Maximum distance of a block pair that can be extended.
    S : int
        Block size in records.
    attributes : list of str
        Attribute names used for pairwise similarity comparisons.
    similarity_threshold : float, default=0.7
        Minimum normalized Damerau-Levenshtein similarity required for each attribute.
    compare_records : callable, default=None
        Optional comparison function receiving two original record rows and returning a boolean.
    true_duplicates : iterable of tuple, default=None
        Known duplicate pairs used for evaluation and precision/recall tracking.
    pSize : int, default=None
        Optional sorting buffer size. When omitted, a default value is inferred from ``N``.
    transitive_closure : bool, default=False
        If ``True``, newly found duplicate pairs are fed into an incremental transitive closure.
    profile : bool, default=True
        Collect per-result runtime, comparison, precision, and recall metrics. Disable to reduce
        evaluation overhead when only duplicate matches are needed.

    Notes
    -----
    The method supports both a single key and multiple keys for attribute concurrency. Block
    pairs are indexed as ``(i, j)`` with ``i <= j``. Following Algorithm 2, a pair is extended
    when ``j - i < R``; its extensions can therefore be compared at distance ``R``.
    """

    def __init__(
        self,
        D,
        K,
        R,
        S,
        attributes,
        true_duplicates=None,
        pSize=None,
        transitive_closure=False,
        profile=True,
        similarity_threshold=0.7,
        compare_records=None,
    ):
        if R < 1:
            raise ValueError("R must be at least 1")
        if S < 1:
            raise ValueError("S must be at least 1")
        self.D = D
        self.K = K if isinstance(K, list) else [K]
        self.R = R
        self.S = S
        self.attributes = attributes
        self.user_pSize = pSize
        self._init_evaluation(true_duplicates, profile, similarity_threshold, compare_records)

        self.transitive_closure = transitive_closure
        self.closure = None

        self._seen = set()
        self._match_ids = set()
        self._N = 0

    def _calc_partition_size(self, N):
        if self.user_pSize is not None:
            return self.user_pSize
        return max(self.S + 1, min(N, max(200, N // max(1, N // 500))))

    def _try_pair(self, index_list, r1, r2, cols, count_known):
        """Compare the records at positions ``r1`` and ``r2``.

        Returns whether the pair counts as a duplicate for the block pair being
        evaluated. A duplicate may have been executed now, or may be a pair already
        found by another key during initialisation when ``count_known=True``.
        """
        if r1 > r2:
            r1, r2 = r2, r1
        code = r1 * self._N + r2

        if code in self._seen:
            return count_known and code in self._match_ids

        self._seen.add(code)
        if not self._compare_positions(r1, r2, cols):
            return False

        self._match_ids.add(code)
        self._record_match(index_list, r1, r2)
        return True

    def _compare_block_pair(self, index_list, b1, b2, cols, count_known, same_block=False):
        """Compare two blocks; returns (duplicates, comparisons) for the density.

        A diagonal block pair (i, i) compares every unordered record pair of the block once,
        a pair of different blocks compares each record of b1 with each record of b2.
        """
        dup_count = 0
        comp_count = 0
        if same_block:
            for x in range(len(b1)):
                for y in range(x + 1, len(b1)):
                    comp_count += 1
                    dup_count += self._try_pair(index_list, b1[x], b1[y], cols, count_known)
        else:
            for r1 in b1:
                for r2 in b2:
                    comp_count += 1
                    dup_count += self._try_pair(index_list, r1, r2, cols, count_known)
        return dup_count, comp_count

    @staticmethod
    def _density(dup_count, comp_count):
        return dup_count / comp_count if comp_count else 0.0

    def fit(self):
        """Fit the progressive-blocking algorithm to the input data and return ``self``."""
        self._start_run()
        self._seen, self._match_ids = set(), set()
        self.closure = IncrementalTransitiveClosure() if self.transitive_closure else None

        N = len(self.D)
        self._N = N
        if N < 2:
            self._finish_run()
            return self

        cols = self._prepare_columns()

        S = self.S
        pSize = self._calc_partition_size(N)
        if pSize < S:
            raise ValueError(f"pSize must be at least S={S}")
        bNum = int(np.ceil(N / S))
        bPerP = max(1, pSize // S)
        partition_size = bPerP * S
        batch_size = bPerP // 4
        if batch_size == 0:
            raise ValueError("pSize must fit at least 4 blocks for progressive expansion")

        num_keys = len(self.K)
        orders, index_lists = {}, {}
        bPairs = []
        tie = itertools.count()
        executed = set()

        def block(k, b):
            return orders[k][b * S : min(N, (b + 1) * S)]

        for k in range(num_keys):
            magpie = MagpieSort(self.D, self.K[k], partition_size, W=1)
            next_block = [0]

            def init_blocks(p, offset, order, plen, known, k=k, magpie=magpie, next_block=next_block):
                orders[k] = order
                while next_block[0] < bNum:
                    b = next_block[0]
                    if min(N, (b + 1) * S) > known:
                        break
                    blk = order[b * S : min(N, (b + 1) * S)]
                    dup_count, c_num = self._compare_block_pair(
                        magpie.index_list,
                        blk,
                        blk,
                        cols,
                        count_known=True,
                        same_block=True,
                    )
                    heapq.heappush(
                        bPairs,
                        (-self._density(dup_count, c_num), 0, next(tie), b, b, k),
                    )
                    executed.add((b, b, k))
                    next_block[0] += 1

            order, index_list, _, _ = magpie.fit(on_partition=init_blocks)
            orders[k], index_lists[k] = order, index_list

        while bPairs:
            bestBPs = []
            while bPairs and len(bestBPs) < batch_size:
                bestBPs.append(heapq.heappop(bPairs))

            pBPs = []
            for _, _, _, i, j, k in bestBPs:
                if j - i >= self.R:
                    continue
                for ni, nj in ((i - 1, j), (i, j + 1)):
                    if ni >= 0 and nj < bNum and (ni, nj, k) not in executed:
                        executed.add((ni, nj, k))
                        pBPs.append((ni, nj, k))

            for ni, nj, k in pBPs:
                dup_count, c_num = self._compare_block_pair(
                    index_lists[k], block(k, ni), block(k, nj), cols, count_known=False
                )
                heapq.heappush(
                    bPairs,
                    (-self._density(dup_count, c_num), nj - ni, next(tie), ni, nj, k),
                )

        self._finish_run()
        return self

    def fit_predict(self):
        """Fit the algorithm and return the discovered duplicate pairs."""
        self.fit()
        return self.discovered_matches
