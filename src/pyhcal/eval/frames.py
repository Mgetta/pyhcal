"""Paired observed/simulated frames: aggregation, filters and partitions.

A PairedFrame holds observed and simulated values on the same timestamps. Its
``view`` says which transforms suit it:

* ``hourly``  - continuous hourly data (hydrograph shape, peaks);
* ``daily``   - continuous daily data, or the daily view of an hourly frame
  built once with ``to_daily()``;
* ``samples`` - discrete samples (grab samples rounded to the hour). Samples
  are compared one by one and never aggregated.

Aggregation always keeps every window and records how complete it is:
``n_values`` (rows from the level below) and ``coverage`` (the share of the
base timesteps present). Coverage adds up through levels: each row counts for
the base timesteps it spans, so hours -> year, hours -> days -> year and
days -> months -> year all give the same figure. Deciding which windows are
complete enough is left to the caller (``filter_by_coverage``).

Rows where observed or simulated is missing are dropped when a frame is
created, so both sides always cover the same timestamps.
"""

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Self, Sequence

import numpy as np
import pandas as pd

from .exceptions import FrameError


MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
SEASONS = {'winter': (12, 1, 2), 'spring': (3, 4, 5), 'summer': (6, 7, 8), 'fall': (9, 10, 11)}

TIMESTEP_FREQ = {"hourly": "h", "daily": "D"}
FREQ_RANK = {"h": 0, "D": 1, "MS": 2, "YS": 3}    # fine to coarse
SUPPORT_KINDS = ("continuous", "discrete", "estimated_continuous")
VIEWS = ("hourly", "daily", "samples")

LABEL = "label"            # group column added by a partition
N_VALUES = "n_values"      # rows from the level below behind each aggregated row
COVERAGE = "coverage"      # share of the base timesteps present behind each row
RESERVED = (LABEL, N_VALUES, COVERAGE)


@dataclass(frozen=True)
class PairedFrame:
    data: pd.DataFrame          # columns: observed, simulated [, observed_flow, ...]
    location_id: str
    constituent: str
    timestep: str               # "hourly" | "daily": the frame's base timestep
    units: str
    quantity_kind: str = "intensive"      # "extensive" allows sum aggregation
    support_kind: str = "continuous"      # "discrete" | "estimated_continuous"

    index_kind: str = "time"              # "time" | "exceedance"
    freq: Optional[str] = None            # current index frequency; set from timestep, None after exceedance

    def __post_init__(self):
        if self.timestep not in TIMESTEP_FREQ:
            raise FrameError(
                f"Unknown timestep '{self.timestep}'; expected one of {tuple(TIMESTEP_FREQ)}"
            )
        if self.support_kind not in SUPPORT_KINDS:
            raise FrameError(
                f"Unknown support_kind '{self.support_kind}'; expected one of {SUPPORT_KINDS}"
            )
        missing = [c for c in ("observed", "simulated") if c not in self.data.columns]
        if missing:
            raise FrameError(f"Frame data needs columns {missing}")
        unpaired = self.data[["observed", "simulated"]].isna().any(axis=1)
        if unpaired.any():
            object.__setattr__(self, "data", self.data[~unpaired.to_numpy()])
        if self.freq is None and self.index_kind == "time":
            object.__setattr__(self, "freq", TIMESTEP_FREQ[self.timestep])

    def replace(self, **kw) -> Self:
        return dataclasses.replace(self, **kw)

    @property
    def view(self) -> str:
        """``samples`` for discrete data, otherwise the timestep (``hourly`` | ``daily``)."""
        if self.support_kind == "discrete":
            return "samples"
        return self.timestep

    @property
    def is_aggregated(self) -> bool:
        return self.index_kind == "time" and self.freq != TIMESTEP_FREQ[self.timestep]

    @property
    def n_points(self) -> int:
        return len(self.data)

    @property
    def is_partitioned(self) -> bool:
        return LABEL in self.data.columns

    @property
    def observed(self) -> np.ndarray:
        return self.data["observed"].to_numpy(float)

    @property
    def simulated(self) -> np.ndarray:
        return self.data["simulated"].to_numpy(float)

    @property
    def frame_key(self) -> tuple[str, str, str, str]:
        return (self.location_id, self.constituent, self.timestep, self.units)

    def groups(self) -> list[tuple[str, Self]]:
        """(label, frame) for each group, in label order; ("all", self) when unpartitioned.

        Fixed label sets (months, seasons, bins) include groups with no rows.
        Group frames drop the label column.
        """
        if not self.is_partitioned:
            return [("all", self)]
        labels = self.data[LABEL]
        rest = self.data.drop(columns=LABEL)
        return [(str(label), self.replace(data=rest[(labels == label).to_numpy()]))
                for label in labels.cat.categories]

    def _with_labels(self, data: pd.DataFrame, labels, categories: Sequence[str]) -> Self:
        if self.is_partitioned:
            raise FrameError("A chain can hold only one partition")
        data = data.assign(**{LABEL: pd.Categorical(labels, categories=list(categories))})
        return self.replace(data=data)

    def _require_unpartitioned(self, step: str) -> None:
        if self.is_partitioned:
            raise FrameError(f"{step} must come before the partition")

    def _column(self, column: str) -> pd.Series:
        if column not in self.data.columns:
            raise FrameError(
                f"Frame has no column '{column}'; available: {list(self.data.columns)}"
            )
        return self.data[column]

    # --- Filters: drop rows, keep every column ---

    def filter_by_month(self, months: Iterable[int]) -> Self:
        if self.index_kind != "time":
            raise FrameError("filter_by_month requires a time index")
        mask = self.data.index.month.isin(tuple(months))
        return self.replace(data=self.data[mask])

    def filter_by_scalar(self, column: str, ge: float = None, le: float = None) -> Self:
        col = self._column(column)
        if ge is None:
            ge = col.min()
        if le is None:
            le = col.max()

        mask = (col >= ge) & (col <= le)
        mask = mask.fillna(False)
        return self.replace(data=self.data[mask])

    def filter_by_quantile(self, column: str, ge: float = None, le: float = None) -> Self:
        if ge is not None:
            ge = float(self._column(column).quantile(ge))

        if le is not None:
            le = float(self._column(column).quantile(le))
        return self.filter_by_scalar(column, ge, le)

    def filter_by_coverage(self, min_coverage: Optional[float]) -> Self:
        """Drop aggregated rows whose coverage is below ``min_coverage``.

        Frames without a coverage column (data at its base timestep) and
        ``min_coverage=None`` are returned unchanged.
        """
        if min_coverage is None or COVERAGE not in self.data.columns:
            return self
        if not 0 < min_coverage <= 1:
            raise FrameError(f"min_coverage must be in (0, 1]; got {min_coverage}")
        return self.replace(data=self.data[self.data[COVERAGE] >= min_coverage - 1e-9])

    # --- Aggregation ---

    def resample(self, freq: str, statistic: str) -> Self:
        """Aggregate observed and simulated into ``freq`` windows.

        Every window holding at least one value is kept, with ``n_values``
        (rows behind it) and ``coverage`` (share of the base timesteps
        present) columns. Other columns are carried along as window means.
        """
        self._require_unpartitioned("resample")
        if self.view == "samples":
            raise FrameError("Samples are compared one by one and are never aggregated")
        if self.index_kind != "time":
            raise FrameError("resample must come before exceedance")
        if freq not in FREQ_RANK:
            raise FrameError(f"Unsupported frequency '{freq}'; use one of {tuple(FREQ_RANK)}")
        if FREQ_RANK[freq] <= FREQ_RANK[self.freq]:
            raise FrameError(f"Cannot resample {self.freq} data to {freq}; it must be coarser")
        if statistic == "sum" and self.quantity_kind != "extensive":
            raise FrameError(f"Cannot sum '{self.quantity_kind}' quantity")

        source = self.data.drop(columns=[c for c in RESERVED if c in self.data.columns])
        row_coverage = self.data[COVERAGE] if COVERAGE in self.data.columns else pd.Series(1.0, index=self.data.index)
        windows = source.resample(freq, label="left", closed="left")
        counts = windows["observed"].count()

        data = windows[["observed", "simulated"]].agg(statistic)
        extra = [c for c in source.columns if c not in ("observed", "simulated")]
        if extra:
            data = data.join(windows[extra].mean())
        data[N_VALUES] = counts
        if len(counts):
            # weight each row by the base timesteps it spans so coverage is exact for any chain
            present = row_coverage * self._base_steps(self.data.index, self.freq)
            covered = present.resample(freq, label="left", closed="left").sum().reindex(counts.index, fill_value=0)
            data[COVERAGE] = covered / self._base_steps(counts.index, freq)
        else:
            data[COVERAGE] = pd.Series(dtype=float)
        data = data[counts > 0].dropna(subset=["observed", "simulated"])
        return self.replace(data=data, freq=freq)

    def _base_steps(self, starts: pd.DatetimeIndex, freq: str) -> pd.Series:
        """Base timesteps in each ``freq`` window starting at ``starts``."""
        base = TIMESTEP_FREQ[self.timestep]
        if freq == base or len(starts) == 0:
            return pd.Series(1.0, index=starts)
        end = starts.max() + pd.tseries.frequencies.to_offset(freq)
        steps = pd.date_range(starts.min(), end, freq=base, inclusive="left")
        sizes = pd.Series(1.0, index=steps).resample(freq, label="left", closed="left").sum()
        return sizes.reindex(starts, fill_value=0)

    def to_daily(self) -> Self:
        """The daily view of an hourly frame: daily means, with hours per day recorded.

        Every day with at least one hour is kept; ``n_values`` is the hours
        behind it and ``coverage`` their share of 24. The result is a daily
        frame at its own base timestep.
        """
        if self.view != "hourly":
            raise FrameError(f"to_daily needs an hourly continuous frame; this frame's view is '{self.view}'")
        return self.resample("D", "mean").replace(timestep="daily")

    def exceedance(self) -> Self:
        self._require_unpartitioned("exceedance")
        n = len(self.data)
        prob = np.arange(1, n + 1) / (n + 1)
        data = pd.DataFrame(
            {
                "observed": np.sort(self.observed)[::-1],
                "simulated": np.sort(self.simulated)[::-1],
            },
            index=prob,
        )
        return self.replace(data=data, index_kind="exceedance", freq=None)

    # --- Partitions: add a LABEL column; the last reshaping step (filters may follow).
    # A quantile filter after a partition uses the whole frame. ---

    def partition_by_month_groups(self, groups: Mapping[str, Iterable[int]]) -> Self:
        """Label rows by month group; rows in no group are dropped."""
        if self.index_kind != "time":
            raise FrameError("Month partitioning requires a time index")
        if self.freq not in ("h", "D", "MS"):
            raise FrameError(
                f"Month partitioning needs sub-annual time data; frame is {self.freq}"
            )
        month_label = {m: label for label, months in groups.items() for m in months}
        labels = self.data.index.month.map(month_label)
        keep = labels.notna()
        return self._with_labels(self.data[keep], labels[keep], groups.keys())

    def partition_by_month(self) -> Self:
        month_map = {name: (i,) for i, name in enumerate(MONTHS, start=1)}
        return self.partition_by_month_groups(month_map)

    def partition_by_season(self) -> Self:
        return self.partition_by_month_groups(SEASONS)

    def partition_by_year(self) -> Self:
        """Label rows by calendar year; only years with data become groups."""
        if self.index_kind != "time":
            raise FrameError("'year' partition requires a time index")
        labels = self.data.index.year.astype(str)
        return self._with_labels(self.data, labels, sorted(set(labels)))

    def partition_by_year_month(self) -> Self:
        """Label rows by month window (``YYYY-MM``); only windows with data become groups."""
        if self.index_kind != "time":
            raise FrameError("'year_month' partition requires a time index")
        if self.freq not in ("h", "D", "MS"):
            raise FrameError(
                f"Month-window partitioning needs sub-annual time data; frame is {self.freq}"
            )
        labels = self.data.index.strftime("%Y-%m")
        return self._with_labels(self.data, labels, sorted(set(labels)))

    def _binned(self, values: np.ndarray, edges: Sequence[float], labels: Sequence[str]) -> Self:
        if len(edges) != len(labels) + 1:
            raise FrameError("Number of bin edges must equal len(labels) + 1")
        if np.any(np.diff(edges) <= 0):
            raise FrameError(f"Bin edges must increase; got {tuple(edges)}")
        keep = ~np.isnan(values) & (values >= edges[0]) & (values <= edges[-1])
        ids = np.digitize(values[keep], edges[1:-1], right=True)
        return self._with_labels(self.data[keep], np.asarray(labels)[ids], labels)

    def partition_by_exceedance_bins(self, edges: Sequence[float], labels: Sequence[str]) -> Self:
        """Label ranked rows by exceedance-probability bin (after ``exceedance()``)."""
        if self.index_kind != "exceedance":
            raise FrameError("exceedance bins require transform = 'fdc'")
        return self._binned(self.data.index.to_numpy(float), edges, labels)

    def partition_by_bins(self, column: str, edges: Sequence[float], labels: Sequence[str]) -> Self:
        """Label rows by the bin ``column`` falls in; bins include their upper edge.

        Rows with a missing value or one outside the edges are dropped.
        """
        return self._binned(self._column(column).to_numpy(float), edges, labels)
