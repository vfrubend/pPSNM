"""Incremental transitive-closure utility."""

from typing import Any


class IncrementalTransitiveClosure:
    """Maintain duplicate clusters under incremental transitive-closure updates.

    Each record identifier is associated with a cluster set. When a duplicate pair is added,
    the two clusters are merged. The smaller cluster is absorbed into the larger one, which
    keeps the total number of pointer updates sub-linear in the number of records.

    Notes
    -----
    The cost of ``add`` is dominated by the number of implied pairs that must be emitted,
    which is inherent to the transitive closure itself.
    """

    def __init__(self):
        self._cluster = {}

    @staticmethod
    def _pair(x, y):
        return (x, y) if x <= y else (y, x)

    def add(self, a: Any, b: Any) -> list[tuple[Any, Any]]:
        """Add a duplicate pair and return all newly implied pairs.

        Parameters
        ----------
        a, b : Any
            Record identifiers that are known to be duplicates.

        Returns
        -------
        list of tuple
            Newly discovered implied duplicate pairs, each normalised as ``(min, max)``.
        """
        if a == b:
            return []
        ca, cb = self._cluster.get(a), self._cluster.get(b)

        if ca is None and cb is None:
            c = {a, b}
            self._cluster[a] = self._cluster[b] = c
            return [self._pair(a, b)]
        if ca is None:
            new = [self._pair(a, m) for m in cb]
            cb.add(a)
            self._cluster[a] = cb
            return new
        if cb is None:
            new = [self._pair(b, m) for m in ca]
            ca.add(b)
            self._cluster[b] = ca
            return new
        if ca is cb:
            return []

        small, large = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
        new = [self._pair(x, y) for x in small for y in large]
        for x in small:
            self._cluster[x] = large
        large |= small
        return new

    def same_cluster(self, a: Any, b: Any) -> bool:
        """Return whether two record ids lie in the same duplicate cluster."""
        ca = self._cluster.get(a)
        return ca is not None and ca is self._cluster.get(b)

    def clusters(self) -> list[set[Any]]:
        """Return all current duplicate clusters, as sets of record ids."""
        return list({id(c): c for c in self._cluster.values()}.values())
