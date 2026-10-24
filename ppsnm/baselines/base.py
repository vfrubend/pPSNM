"""Shared implementation for non-progressive baselines."""

import numpy as np

from ..core.evaluation import ProgressiveEvaluation
from ..core.magpie_sort import _factorized_key_codes


class _BaselineBase(ProgressiveEvaluation):
    """Common machinery for the non-progressive baseline algorithms.

    Notes
    -----
    The baselines intentionally omit the paper's progressive contributions, so they
    represent the state of the art before it:

    * the sort is complete before the first comparison (no MagpieSort);
    * no look-ahead and no adaptive ordering of comparisons;
    * no memory of executed comparisons: every pass is an independent run, as in the
      traditional multi-pass method, and the passes run one after another in the given key
      order (no attribute concurrency);
    * the transitive closure is a blocking step after all passes (traditional duplicate
      detection evaluates the closure only after the final pass), not an incremental one.

    Window semantics remain identical to PSNM: a window of size ``W`` covers rank distances
    ``1..W-1``. For a single key, all algorithms execute the same comparisons and only differ
    in the order in which they are executed.

    Set ``profile=False`` to skip runtime, comparison-count, and quality logging when only the
    discovered matches are needed.
    """

    def __init__(
        self,
        D,
        K,
        W,
        attributes,
        true_duplicates=None,
        transitive_closure=False,
        profile=True,
        similarity_threshold=0.7,
        compare_records=None,
    ):
        self.D = D
        self.K = K if isinstance(K, list) else [K]
        self.W = W
        self.attributes = attributes
        self._init_evaluation(true_duplicates, profile, similarity_threshold, compare_records)

        self.transitive_closure = transitive_closure

    def _sort_order(self, key):
        """Return row ids sorted by ``key``.

        The ordering is complete and stable, with missing values placed last.
        This matches the total order MagpieSort converges to, so all algorithms
        see identical sorted lists.
        """
        codes = _factorized_key_codes(self.D, key)
        return np.argsort(codes, kind="stable")

    def _compare(self, order, index_list, a, b, cols):
        """Compare the records at sort ranks a and b."""
        r1, r2 = int(order[a]), int(order[b])
        if self.profile_enabled:
            self._comps += 1
        if self.compare_fast(r1, r2, cols):
            l1, l2 = index_list[r1], index_list[r2]
            self._record_pair((l1, l2) if l1 <= l2 else (l2, l1))

    def _close(self):
        """Blocking transitive closure of the final result."""
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in list(self.discovered_matches):
            parent[find(a)] = find(b)
        clusters = {}
        for x in list(parent):
            clusters.setdefault(find(x), []).append(x)
        for members in clusters.values():
            members.sort()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    self._record_pair((members[i], members[j]))

    def _traverse(self, order, index_list, cols):
        raise NotImplementedError

    def fit(self):
        """Fit the baseline algorithm to the input data and return ``self``."""
        self._start_run()
        N = len(self.D)
        if N < 2 or self.W < 2:
            self._finish_run()
            return self

        cols = self._prepare_columns()
        index_list = self.D.index.to_numpy()
        for key in self.K:
            order = self._sort_order(key)
            self._traverse(order, index_list, cols)

        if self.transitive_closure:
            self._close()

        self._finish_run()
        return self

    def fit_predict(self):
        """Fit the algorithm and return the discovered duplicate pairs."""
        self.fit()
        return self.discovered_matches

    def run(self):
        """Backward-compatible alias for ``fit``."""
        self.fit()
        return self.log
