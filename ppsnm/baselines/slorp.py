"""Sorted List of Record Pairs baseline."""

from .base import _BaselineBase


class SortedListOfRecordPairs(_BaselineBase):
    """SLORP hint (Whang et al.), processed partition by partition (memory-bounded).

    "...a small window of size two that quickly finds the most promising records. This static
    approach has already been proposed as the Sorted List of Record Pairs hint" (paper, Sec. 3).

    Only one partition is in memory at a time. The partitions are consecutive ranges of the
    sorted order with an overlap of W-1 records, like PSNM's, so the window can slide across
    boundaries and all algorithms execute the same comparisons. A pair inside the overlap is
    processed by the first partition that contains it (a structural rule, no memory of executed
    comparisons).

    complete_partitions selects how the partitions are visited:
      False (default): the hint is calculated over *all* partitions (Whang et al. [1], "partly
          addresses" the problem, paper Sec. 1): rank-distance 1 on every partition, then
          rank-distance 2 on every partition, ... Partitions are loaded again for every distance,
          which is why the paper gives SLORP partition caching "for fairness" (Sec. 8.3). This is
          the setting of the paper's evaluation.
      True: the hint is calculated per partition: all rank-distances on one partition, then the
          next partition. "By completing one partition of a large dataset after another, the
          overall duplicate detection process is no longer progressive" (paper, Sec. 1).

    pSize: partition size in records, default as in PSNM (must be >= W).
    profile: whether to collect evaluation metrics, default as in the other algorithms.
    """

    def __init__(
        self,
        D,
        K,
        W,
        attributes,
        true_duplicates=None,
        transitive_closure=False,
        pSize=None,
        complete_partitions=False,
        profile=True,
        similarity_threshold=0.7,
        compare_records=None,
    ):
        super().__init__(
            D,
            K,
            W,
            attributes,
            true_duplicates,
            transitive_closure,
            profile,
            similarity_threshold,
            compare_records,
        )
        self.user_pSize = pSize
        self.complete_partitions = complete_partitions

    def _partition_size(self, N):
        if self.user_pSize is not None:
            if self.user_pSize < self.W:
                raise ValueError(f"pSize must be at least W={self.W}")
            return self.user_pSize
        return max(self.W + 1, min(N, max(100, N // max(1, N // 500))))  # same default as PSNM

    def _traverse(self, order, index_list, cols):
        N = len(order)
        pSize = self._partition_size(N)
        step = max(1, pSize - self.W + 1)
        parts = []  # (offset, end, done_below)
        offset = 0
        while True:
            end = min(N, offset + pSize)
            # pairs ending below done_below lie in the overlap and belong to the previous partition
            parts.append((offset, end, offset + self.W - 1 if offset > 0 else 0))
            if end == N:
                break
            offset += step

        def process(part, d_from, d_to):
            offset, end, done_below = part
            for dist in range(d_from, d_to):
                for a in range(max(offset, done_below - dist), end - dist):
                    self._compare(order, index_list, a, a + dist, cols)

        max_dist = min(self.W - 1, N - 1)
        if self.complete_partitions:
            for part in parts:
                process(part, 1, max_dist + 1)
        else:
            for dist in range(1, max_dist + 1):
                for part in parts:
                    process(part, dist, dist + 1)
