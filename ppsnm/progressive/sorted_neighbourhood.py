"""Progressive sorted-neighbourhood method."""

from __future__ import annotations

import numpy as np

from ..core.evaluation import ProgressiveEvaluation
from ..core.incremental_transitive_closure import IncrementalTransitiveClosure
from ..core.magpie_sort import MagpieSort


class ProgressiveSortedNeighbourhoodMethod(ProgressiveEvaluation):
    """Progressive Sorted Neighbourhood Method (PSNM) with attribute concurrency.

    Parameters
    ----------
    D : pandas.DataFrame
        Input records.
    K : str or list of str
        Sorting key, or several keys for attribute concurrency.
    W : int
        Window size covering rank distances ``1..W-1``.
    I : int
        Number of rank distances processed per progressive iteration.
    attributes : list of str
        Attribute names used to compare record pairs.
    similarity_threshold : float, default=0.7
        Minimum normalized Damerau-Levenshtein similarity required for each attribute.
    compare_records : callable, default=None
        Optional comparison function receiving two original record rows and returning a boolean.
    true_duplicates : iterable of tuple, default=None
        Optional known duplicate pairs used for evaluation.
    pSize : int, default=None
        Optional progressive-sort buffer size.
    transitive_closure : bool, default=False
        If ``True``, incremental transitive closure is applied as matches are discovered.
    profile : bool, default=True
        Collect per-result runtime, comparison, precision, and recall metrics. Disable to reduce
        evaluation overhead when only duplicate matches are needed.

    Notes
    -----
    The window size ``W`` matches the traditional sorted-neighbourhood setup, while the
    progressive iterator controls how many rank distances are explored per pass. The base
    evaluator keeps the logging, runtime accounting, and quality metrics.
    """

    def __init__(
        self,
        D,
        K,
        W,
        I,
        attributes,
        true_duplicates=None,
        pSize=None,
        transitive_closure=False,
        profile=True,
        similarity_threshold=0.7,
        compare_records=None,
    ):
        if W < 1:
            raise ValueError("W must be at least 1")
        if I < 1:
            raise ValueError("I must be at least 1")
        if pSize is not None and pSize < W:
            raise ValueError(f"pSize must be at least W={W}")
        self.D = D
        self.K = K if isinstance(K, list) else [K]
        self.W, self.I, self.attributes = W, I, attributes
        self.user_pSize = pSize
        self._init_evaluation(true_duplicates, profile, similarity_threshold, compare_records)

        self.transitive_closure = transitive_closure
        self.closure = None

        self._seen = set()
        self._match_ids = set()
        self._N = 0

    def _try_pair(self, order, index_list, a, b, cols, counted):
        """Compare the records at sort ranks a < b of one key's order.

        `counted` is None in the main loop (only duplicates found by this very comparison
        count for the key). During initialisation it is a per-key set: duplicates already
        found through another key count as well, once each (paper, Sec. 5.1).

        Returns (counts, expand):
          counts: whether this pair counts toward the key's duplicate count
          expand: the comparison was executed now and matched -> run look-ahead
        """
        r1, r2 = int(order[a]), int(order[b])
        if r1 > r2:
            r1, r2 = r2, r1
        code = r1 * self._N + r2

        if code in self._seen:
            if counted is not None and code in self._match_ids and code not in counted:
                counted.add(code)
                return True, False
            return False, False

        self._seen.add(code)
        if not self._compare_positions(r1, r2, cols):
            return False, False

        self._match_ids.add(code)
        if counted is not None:
            counted.add(code)
        self._record_match(index_list, r1, r2)
        return True, True

    def _look_ahead(self, order, index_list, a, b, cols, counted, known, pending=None):
        """Iteratively explore (a+1, b) and (a, b+1) around a duplicate (a, b).

        `known` is the number of final sort ranks (during progressive sorting only a prefix
        of the order exists); look-ahead never goes beyond it.
        """
        found = 0
        max_dist = self.W - 1
        stack = [(a, b)]
        while stack:
            ca, cb = stack.pop()
            for na, nb in ((ca + 1, cb), (ca, cb + 1)):
                if na >= nb or nb - na > max_dist:
                    continue
                if nb >= known:
                    if pending is not None:
                        pending.add((ca, cb))
                    continue
                counts, expand = self._try_pair(order, index_list, na, nb, cols, counted)
                found += counts
                if expand:
                    stack.append((na, nb))
        return found

    def _step_partition(
        self,
        order,
        index_list,
        offset,
        plen,
        cols,
        d_from,
        d_to,
        counted,
        known,
        pending=None,
    ):
        """Rank-distances d_from..d_to-1 on the partition covering ranks offset..offset+plen-1."""
        new_count = 0
        for dist in range(d_from, d_to):
            for i in range(plen - dist):
                a, b = offset + i, offset + i + dist
                counts, expand = self._try_pair(order, index_list, a, b, cols, counted)
                new_count += counts
                if expand:
                    new_count += self._look_ahead(order, index_list, a, b, cols, counted, known, pending)
        return new_count

    def _step(self, order, index_list, partitions, step_size, cols, d_from, d_to, counted):
        """One progressive iteration on all partitions of a (fully sorted) key."""
        new_count = 0
        for p, part in partitions.items():
            new_count += self._step_partition(
                order,
                index_list,
                p * step_size,
                len(part),
                cols,
                d_from,
                d_to,
                counted,
                len(order),
            )
        return new_count

    def fit(self):
        """Fit the PSNM model to the input data and return ``self``."""
        self._start_run()
        self._seen, self._match_ids = set(), set()
        self.closure = IncrementalTransitiveClosure() if self.transitive_closure else None

        N = len(self.D)
        self._N = N
        if N < 2 or self.W < 2:
            self._finish_run()
            return self

        cols = self._prepare_columns()

        num_keys = len(self.K)
        pSize = (
            self.user_pSize if self.user_pSize is not None else max(self.W + 1, min(N, max(100, N // max(1, N // 500))))
        )
        step_size = max(1, pSize - self.W + 1)

        orders, index_lists, partitions = {}, {}, {}
        next_dist = np.ones(num_keys, dtype=int)
        d_counts = np.zeros(num_keys, dtype=int)

        for k in range(num_keys):
            magpie = MagpieSort(self.D, self.K[k], pSize, W=self.W)
            end = min(next_dist[k] + self.I, self.W)
            counted = set()
            count = [0]
            pending_lookahead = set()

            def first_iteration(
                p,
                offset,
                order,
                plen,
                known,
                magpie=magpie,
                end=end,
                counted=counted,
                count=count,
                pending=pending_lookahead,
            ):
                for a, b in tuple(pending):
                    pending.remove((a, b))
                    count[0] += self._look_ahead(order, magpie.index_list, a, b, cols, counted, known, pending)
                count[0] += self._step_partition(
                    order,
                    magpie.index_list,
                    offset,
                    plen,
                    cols,
                    1,
                    end,
                    counted,
                    known,
                    pending,
                )

            order, index_list, parts, _ = magpie.fit(on_partition=first_iteration)
            orders[k], index_lists[k], partitions[k] = order, index_list, parts
            d_counts[k] = count[0]
            next_dist[k] = end

        while True:
            open_keys = [k for k in range(num_keys) if next_dist[k] < self.W]
            if not open_keys:
                break
            k = max(open_keys, key=lambda x: (d_counts[x], -x))

            end = min(next_dist[k] + self.I, self.W)
            d_counts[k] = self._step(
                orders[k],
                index_lists[k],
                partitions[k],
                step_size,
                cols,
                next_dist[k],
                end,
                None,
            )
            next_dist[k] = end

        self._finish_run()
        return self

    def fit_predict(self):
        """Fit the algorithm and return the discovered duplicate pairs."""
        self.fit()
        return self.discovered_matches
