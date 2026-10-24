"""Shared evaluation and comparison utilities."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from rapidfuzz.distance import DamerauLevenshtein


class ProgressiveEvaluation:
    """Shared evaluation machinery for all progressive duplicate-detection methods.

    This base class owns the data cleaning, similarity check, logging, and progress-quality
    metrics used across PSNM, PB, SNM, and SLORP. Concrete algorithms only need to define how
    candidate pairs are generated and in which order they are explored.

    Notes
    -----
    Derived algorithms should call ``_init_evaluation(true_duplicates)`` in their
    constructor and wrap their execution in ``_start_run`` / ``_finish_run``.
    """

    similarity_threshold = 0.7

    def _init_evaluation(
        self,
        true_duplicates=None,
        profile=True,
        similarity_threshold=0.7,
        compare_records: Callable[[pd.Series, pd.Series], bool] | None = None,
    ):
        if not self.D.index.is_unique:
            raise ValueError("DataFrame index must contain unique record identifiers")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.profile_enabled = profile
        self.similarity_threshold = similarity_threshold
        self.compare_records = compare_records
        self.true_duplicates = (
            {(min(p), max(p)) for p in true_duplicates} if profile and true_duplicates is not None else None
        )
        self.discovered_matches = set()
        self.log = []
        self.runtime = 0.0 if profile else None
        self.total_comparisons = 0 if profile else None
        self._tp = 0
        self._comps = 0
        self._t0 = 0.0

    def _start_run(self):
        """Reset all run state; call first in run()."""
        self.discovered_matches = set()
        self._tp = 0
        self._comps = 0
        self.log = []
        if self.profile_enabled:
            self.runtime, self.total_comparisons = 0.0, 0
            self._t0 = time.perf_counter()
            self.log = [self.get_current_metrics(0.0, 0)]
        else:
            self.runtime = self.total_comparisons = None

    def _finish_run(self):
        """Store the overall cost; call last in run(). Returns the log."""
        if self.profile_enabled:
            self.runtime = time.perf_counter() - self._t0
            self.total_comparisons = self._comps
        return self.log

    @property
    def performance(self):
        """Return profiling samples as a DataFrame with named metric columns."""
        columns = ["elapsed_seconds", "comparisons", "precision", "recall", "duplicates_found"]
        if not self.profile_enabled:
            return pd.DataFrame(columns=columns)
        metrics = pd.DataFrame(
            self.log,
            columns=["elapsed_seconds", "comparisons", "precision", "recall"],
        )
        if self.true_duplicates:
            metrics["duplicates_found"] = np.rint(metrics["recall"] * len(self.true_duplicates)).astype(np.int64)
        else:
            metrics["duplicates_found"] = np.arange(len(metrics), dtype=np.int64)
        return metrics[columns]

    def _require_profile(self):
        if not self.profile_enabled:
            raise RuntimeError("This operation requires profiling; enable profile=True.")

    # --------------------------------------------------------------- comparison
    @staticmethod
    def _clean_str(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if not val:
            return ""
        cleaned = "".join(char for char in str(val).casefold() if char.isalnum() or char.isspace())
        return " ".join(cleaned.split())

    def _prepare_columns(self):
        """Cleaned string columns of all compared attributes, indexed by record position."""
        if self.compare_records is not None:
            return {}
        return {
            attr: np.array([self._clean_str(val) for val in self.D[attr]], dtype=object)
            for attr in set(self.attributes)
        }

    def compare_fast(self, r1_pos, r2_pos, cleaned_col_arrays):
        """Return whether the records at the two positions are a duplicate."""
        if self.compare_records is not None:
            return self.compare_records(self.D.iloc[r1_pos], self.D.iloc[r2_pos])
        for attr in self.attributes:
            v1, v2 = cleaned_col_arrays[attr][r1_pos], cleaned_col_arrays[attr][r2_pos]
            if not v1 or not v2 or DamerauLevenshtein.normalized_similarity(v1, v2) < self.similarity_threshold:
                return False
        return True

    def _compare_positions(self, r1_pos, r2_pos, cleaned_col_arrays):
        """Compare two row positions and count the executed comparison when profiling."""
        if self.profile_enabled:
            self._comps += 1
        return self.compare_fast(r1_pos, r2_pos, cleaned_col_arrays)

    def _record_match(self, index_list, r1_pos, r2_pos):
        """Record a match, including any pairs implied by the optional closure."""
        left, right = index_list[r1_pos], index_list[r2_pos]
        if self.closure is None:
            self._record_pair((left, right) if left <= right else (right, left))
            return
        for pair in self.closure.add(left, right):
            self._record_pair(pair)

    def get_current_metrics(self, elapsed, comp_count):
        """Return the current runtime and quality metrics for the in-flight run."""
        self._require_profile()
        if self.true_duplicates is None:
            return (elapsed, comp_count, 0.0, 0.0)
        num_disc = len(self.discovered_matches)
        tp = self._tp
        return (
            elapsed,
            comp_count,
            tp / num_disc if num_disc else 1.0,
            tp / len(self.true_duplicates) if self.true_duplicates else 0.0,
        )

    def _record_pair(self, pair):
        """Register a newly discovered duplicate pair (record labels, ordered) and log it."""
        if pair in self.discovered_matches:
            return
        self.discovered_matches.add(pair)
        if not self.profile_enabled:
            return
        if self.true_duplicates is not None and pair in self.true_duplicates:
            self._tp += 1
        self.log.append(self.get_current_metrics(time.perf_counter() - self._t0, self._comps))

    @staticmethod
    def weights(weighting, n, L=None):
        """Weighting function w(t) for the interval indices t = 1..n (paper, Fig. 3).

        Every function satisfies Definition 3: values in [0, 1], non-increasing, w(1) = 1.
          economic    : max(1 - (t-1)/budget, 0), the budget being the n intervals (area under the curve)
          recall      : 1 (no preference for early results; allowed by Definition 3, not drawn in Fig. 3)
          fluctuating : min(1/t^2, 1)
          timeslot    : 1 if t <= L else 0, L = length of the timeslot in intervals (required, >= 1)
          pipeline    : 1/(ln(t) + 1). Fig. 3 prints 1/(ln(t+1)+1) for a continuous t starting at 0;
                        for the discrete interval index t = 1..n that gives w(1) = 0.59 and violates
                        Definition 3, so the interval is evaluated at its start, t-1.
        """
        t = np.arange(1, n + 1, dtype=float)
        if weighting == "economic":
            return np.maximum(1.0 - (t - 1.0) / n, 0.0)
        if weighting == "recall":
            return np.ones_like(t)
        if weighting == "fluctuating":
            return np.minimum(1.0 / t**2, 1.0)
        if weighting == "timeslot":
            if L is None or L < 1:
                raise ValueError("timeslot weighting needs a slot length L of at least one interval")
            return (t <= L + 1e-9).astype(float)
        if weighting == "pipeline":
            return 1.0 / (np.log(t) + 1.0)
        raise ValueError(f"unknown weighting: {weighting}")

    def progressive_quality(
        self,
        log=None,
        T=None,
        weighting="economic",
        N=None,
        x="time",
        n_intervals=100,
        interval=None,
        L=1.0,
    ):
        """Progressive quality Q(T) = 1/N * sum_{t=1..T} w(t) * r(t)  (paper Sec. 7.2, Eq. 1).

        log          : defaults to this algorithm's own log from its last run.
        T            : measurement time. Defaults to this run's total runtime (x="time") or total
                       comparisons (x="comparisons"). When comparing algorithms, use the minimum
                       over all of them (Definition 2) - see `q_table`, which does that.
        weighting    : "economic" (= area under the result curve), "recall" (w=1), "fluctuating",
                       "timeslot" (needs L) or "pipeline" -- see `weights` (Fig. 3).
        N            : total number of duplicates; defaults to len(true_duplicates).
        x            : "time" (seconds) or "comparisons" (deterministic, hardware independent).
        n_intervals  : number of equidistant sampling intervals t = 1..n (ignored if interval given).
        interval     : length of one interval in x-units. The non-economic weightings depend on it.
        L            : length of the timeslot as a fixed duration in x-units (seconds or comparisons),
                       independent of T (timeslot weighting only).
        """
        log = self.log if log is None else log
        self._require_profile()
        col = {"time": 0, "comparisons": 1}[x]
        if N is None:
            if self.true_duplicates is None:
                raise ValueError("N is required when no true_duplicates were given")
            N = len(self.true_duplicates)
        if T is None:
            T = self.runtime if x == "time" else self.total_comparisons
        if T <= 0 or N <= 0:
            return 0.0

        if interval is None:
            interval = T / n_intervals
        n = int(np.ceil(T / interval - 1e-9))

        # r(t): duplicates found in interval (t-1, t]; results later than T are ignored.
        rec = np.array([row[3] for row in log], dtype=float)
        xs = np.array([row[col] for row in log[1:]], dtype=float)
        if self.true_duplicates is not None:
            xs = xs[np.diff(rec) > 1e-12]  # keep true positives only
        bins = np.maximum(1, np.ceil(xs / interval - 1e-9)).astype(int)
        bins = bins[bins <= n]
        r = np.bincount(bins, minlength=n + 1)[1 : n + 1]

        w = self.weights(weighting, n, None if L is None else L / interval)
        return float(np.sum(w * r) / N)

    # ------------------------------------------- comparing several algorithms
    @staticmethod
    def common_t(runs, x="time"):
        """Return the common measurement time T for the compared runs.

        This is the smallest overall cost among the algorithms, as in Definition 2
        of the paper.
        """
        attr = "runtime" if x == "time" else "total_comparisons"
        values = [getattr(alg, attr) for alg in runs.values()]
        if any(value is None for value in values):
            raise RuntimeError("Progressive quality requires profiling; enable profile=True.")
        return min(values)

    common_T = common_t

    @staticmethod
    def _gold_size(runs, N):
        if N is not None:
            return N
        for alg in runs.values():
            if alg.true_duplicates is not None:
                return len(alg.true_duplicates)
        raise ValueError("N is required when no algorithm was given true_duplicates")

    @staticmethod
    def q_table(runs, N=None, weighting="economic", x="time", n_intervals=100, T=None, **kw):
        """Return Q(T) for every algorithm at a common measurement budget.

        The same T and sampling interval are used for all runs so the values are
        directly comparable.
        """
        N = ProgressiveEvaluation._gold_size(runs, N)
        T = ProgressiveEvaluation.common_t(runs, x) if T is None else T
        interval = T / n_intervals
        return {
            name: alg.progressive_quality(T=T, weighting=weighting, N=N, x=x, interval=interval, **kw)
            for name, alg in runs.items()
        }

    @staticmethod
    def q_curve(runs, N=None, weighting="economic", x="time", points=40, n_intervals=100, **kw):
        """Return Q as a function of the measurement time T.

        The values are reported for T between 5% and 100% of the common runtime or
        comparison budget, and the result is a tuple ``(Ts, {name: [Q(T), ...]})``.
        """
        N = ProgressiveEvaluation._gold_size(runs, N)
        Tmax = ProgressiveEvaluation.common_t(runs, x)
        Ts = np.linspace(0.05 * Tmax, Tmax, points)
        out = {
            name: [
                alg.progressive_quality(T=T, weighting=weighting, N=N, x=x, interval=T / n_intervals, **kw) for T in Ts
            ]
            for name, alg in runs.items()
        }
        return Ts, out

    @staticmethod
    def plot_q(
        runs,
        N=None,
        x="time",
        weightings=("economic", "recall", "fluctuating", "pipeline"),
        n_intervals=100,
        L=None,
    ):
        """Return a figure with three progressive-quality views.

        The plot compares duplicate discovery over cost, the time-based ``Q(T)``
        curve, and the common-T values across weighting functions.
        """
        import matplotlib.pyplot as plt

        E = ProgressiveEvaluation
        N = E._gold_size(runs, N)
        col = {"time": 0, "comparisons": 1}[x]
        unit = "sec" if x == "time" else "comparisons"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        q_main = E.q_table(runs, N, "economic", x, n_intervals)
        for name, alg in runs.items():
            axes[0].plot(
                [row[col] for row in alg.log],
                np.arange(len(alg.log)),
                label=f"{name} ({q_main[name]:.1%})",
            )
        axes[0].axhline(N, ls="--", c="grey", alpha=0.7, label="Target Duplicates")
        axes[0].set(
            title="Duplicates (economic Q in legend)",
            xlabel="Time [sec]" if x == "time" else "Comparisons [#]",
            ylabel="Duplicates [#]",
        )
        axes[0].legend(loc="lower right")

        Ts, curves = E.q_curve(runs, N, "economic", x, n_intervals=n_intervals)
        for name, qs in curves.items():
            axes[1].plot(Ts, qs, label=name)
        axes[1].set(
            title="Progressive quality Q(T)",
            xlabel=f"Measurement time T [{unit}]",
            ylabel="Q(T) (economic)",
            ylim=(0, 1),
        )
        axes[1].legend()

        tables = {w: E.q_table(runs, N, w, x, n_intervals, L=L) for w in weightings}
        width = 0.8 / len(runs)
        for k, name in enumerate(runs):
            axes[2].bar(
                np.arange(len(weightings)) + k * width,
                [tables[w][name] for w in weightings],
                width,
                label=name,
            )
        axes[2].set_xticks(np.arange(len(weightings)) + width * (len(runs) - 1) / 2, weightings)
        axes[2].set(title="Q at the common T per weighting", ylim=(0, 1))
        axes[2].legend()
        fig.tight_layout()
        return fig
