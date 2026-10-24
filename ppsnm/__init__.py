"""Progressive duplicate detection algorithms.

This package exposes the algorithms from the paper and the shared building blocks used by
all flavours of the implementation.
"""

from .baselines.slorp import SortedListOfRecordPairs
from .baselines.snm import SortedNeighbourhoodMethod
from .core.evaluation import ProgressiveEvaluation
from .core.incremental_transitive_closure import IncrementalTransitiveClosure
from .core.magpie_sort import MagpieSort
from .progressive.blocking import ProgressiveBlocking
from .progressive.sorted_neighbourhood import ProgressiveSortedNeighbourhoodMethod

__all__ = [
    "IncrementalTransitiveClosure",
    "MagpieSort",
    "ProgressiveBlocking",
    "ProgressiveEvaluation",
    "ProgressiveSortedNeighbourhoodMethod",
    "SortedListOfRecordPairs",
    "SortedNeighbourhoodMethod",
]

__version__ = "1.0.0"
