"""Transforms, partitions and comparisons: what is compared, how it is grouped, how it is scored.

Three separate choices make one evaluation:

* a **transform** decides which values exist: a chain of PairedFrame methods
  (resample, filter, rank) that always returns one paired series. Its name
  says what each value is (``annual_mean``: one year's mean);
* a **partition** (``by``) decides which values are compared together:
  ``none``, ``month``, ``season``, ``year``, ``year_month``, ``fdc_segment``,
  ``flow_regime``;
* a **comparison** reduces each group to one number (see comparisons.py).

Each transform lists, per partition, the comparisons that suit it and the
points a group needs (``Grouping``). Results carry ``transform, by, label,
comparison``.

Each transform runs on one view of the data (see frames.py):

* ``hourly``  - hydrograph shape and peaks from continuous hourly data;
* ``daily``   - water balance, seasonal pattern and flow regime. An hourly
  frame provides a daily view through ``to_daily()``;
* ``samples`` - grab samples, each compared with the simulated value for the
  same hour. Samples are grouped, never aggregated.

Two families look at flow regimes and answer different questions:

* ``*_days`` / ``*_hours`` keep the times at which the observed series met a
  condition and compare both series at those times, so timing errors count.
  Selecting on observed extremes pulls the simulated side toward the middle:
  expect some under-prediction at high flows and over-prediction at low flows
  even from a well-calibrated model.
* ``fdc`` ranks each series on its own and pairs values by exceedance
  probability, so timing is ignored.

The ``*_days`` / ``*_hours`` transforms select on the frame's own observed
series, so they are meant for flow frames.

Computing always returns a value when the arithmetic allows. Each result
records what is needed to judge it downstream: ``n_points``,
``required_points`` (the larger of the grouping's and the comparison's
recommendation), ``sufficient``, coverage, and each side's mean. Two choices
can't be made afterwards, because they change which values go into a number;
both are run options that keep everything by default:

* ``min_day_coverage`` - leave days with too few hours out of an hourly
  frame's daily view, before any daily transform runs (so quantile thresholds
  and the FDC never see them). ``MIN_DAY_COVERAGE`` is a recommended value;
* ``min_coverage`` - leave points (days, months, years) with too little
  coverage out of each group before comparing. ``MIN_COVERAGE`` is a
  recommended value.

Choosing comparisons:

* scores (nse, kge, r ...) need values paired in time across their full
  range, so they are left off ranked output and off transforms that pick
  times by the size of the observed value;
* on the whole ranked curve only pointwise comparisons add anything, because
  a statistic of one side is unchanged by ranking;
* a group holding one value (one year, one month window) supports only the
  error between the two values.
"""

import inspect
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from pyhcal.eval.comparisons import COMPARISONS, MARGINALS, POINTWISE
from pyhcal.eval.exceptions import ConfigError
from pyhcal.eval.frames import COVERAGE, SEASONS, PairedFrame


MIN_DAY_COVERAGE = 0.8   # recommended day rule: share of 24 hours a day needs (20 h)
MIN_COVERAGE = 0.8       # recommended point rule: share of a month's or year's timesteps

# Recommended points per group, in each transform's own unit.
MIN_HOURS = 720
MIN_DAYS = 30
MIN_MONTHS = 12
MIN_YEARS = 3
MIN_GROUP_DAYS = 10
MIN_MONTH_DAYS = 24
MIN_YEAR_DAYS = 292
MIN_SAMPLES = 10
MIN_SAMPLES_PER_MONTH = 3
MIN_SAMPLES_PER_GROUP = 5

# Comparisons that suit each kind of output.
ALL = tuple(COMPARISONS)
ERRORS = MARGINALS + POINTWISE
ONE_VALUE = ("error_of_mean", "pct_error_of_mean", "ratio_of_mean")

SPRING_RUNOFF_MONTHS = (3, 4, 5)
SUMMER_LOW_FLOW_MONTHS = (7, 8, 9)
BASEFLOW_FRACTION = "observed_baseflow_fraction"

FDC_SEGMENT_EDGES = (0.0, 0.10, 0.50, 1.0)
FDC_SEGMENT_LABELS = ("high_10", "mid_40", "low_50")

# Exceedance probability of flow at the sample hour within the full flow
# record, supplied with a samples frame. Zones follow EPA load duration curves.
FLOW_EXCEEDANCE = "flow_exceedance"
FLOW_REGIME_EDGES = (0.0, 0.10, 0.40, 0.60, 0.90, 1.0)
FLOW_REGIME_LABELS = ("high", "moist", "mid_range", "dry", "low")


# --- Hourly view: hydrograph shape and peaks ---

def hourly_mean(pf: PairedFrame) -> PairedFrame:
    """Hourly values. Hydrograph shape and storm timing."""
    return pf


def annual_peak(pf: PairedFrame) -> PairedFrame:
    """Largest hourly value of each calendar year on each side. Peak magnitude, whatever its hour."""
    return pf.resample("YS", "max")


def high_flow_hours(pf: PairedFrame) -> PairedFrame:
    """Hours in the top 10% of observed hourly values. Storm peaks and routing."""
    return pf.filter_by_quantile("observed", ge=0.90)


# --- Daily view: overall fit and water balance ---

def daily_mean(pf: PairedFrame) -> PairedFrame:
    """Daily means. Overall fit and total volume."""
    return pf


def monthly_mean(pf: PairedFrame) -> PairedFrame:
    """Mean of each calendar month, one value per month per year. Seasonal timing without daily noise."""
    return pf.resample("MS", "mean")


def annual_mean(pf: PairedFrame) -> PairedFrame:
    """Mean of each calendar year. Annual water balance."""
    return pf.resample("YS", "mean")


def monthly_total(pf: PairedFrame) -> PairedFrame:
    """Total of each calendar month, for extensive quantities such as loads or runoff depth.

    A month with gaps is a partial total on both sides.
    """
    return pf.resample("MS", "sum")


def annual_total(pf: PairedFrame) -> PairedFrame:
    """Total of each calendar year, for extensive quantities such as loads or runoff depth.

    A year with gaps is a partial total on both sides.
    """
    return pf.resample("YS", "sum")


def annual_daily_max(pf: PairedFrame) -> PairedFrame:
    """Largest daily mean of each calendar year on each side. Peak magnitude, whatever its date."""
    return pf.resample("YS", "max")


# --- Daily view, flow regime on observed dates: timing errors count ---

def high_flow_days(pf: PairedFrame) -> PairedFrame:
    """Days in the top 10% of observed daily means. Storm runoff and peak routing."""
    return pf.filter_by_quantile("observed", ge=0.90)


def low_flow_days(pf: PairedFrame) -> PairedFrame:
    """Days in the bottom 50% of observed daily means. Baseflow and recession."""
    return pf.filter_by_quantile("observed", le=0.50)


def stormflow_days(pf: PairedFrame) -> PairedFrame:
    """Days when baseflow is at most half of observed flow. Surface runoff and interflow."""
    return pf.filter_by_scalar(BASEFLOW_FRACTION, le=0.50)


def baseflow_days(pf: PairedFrame) -> PairedFrame:
    """Days when baseflow is at least 80% of observed flow. Groundwater recession."""
    return pf.filter_by_scalar(BASEFLOW_FRACTION, ge=0.80)


def spring_high_flow_days(pf: PairedFrame) -> PairedFrame:
    """March-May days in the top 25% of observed spring daily means. Snowmelt runoff."""
    return (
        pf.filter_by_month(SPRING_RUNOFF_MONTHS)
        .filter_by_quantile("observed", ge=0.75)
    )


def summer_low_flow_days(pf: PairedFrame) -> PairedFrame:
    """July-September days in the bottom 25% of observed daily means for those months.

    Groundwater recession and evapotranspiration.
    """
    return (
        pf.filter_by_month(SUMMER_LOW_FLOW_MONTHS)
        .filter_by_quantile("observed", le=0.25)
    )


def summer_stormflow_days(pf: PairedFrame) -> PairedFrame:
    """June-August days when baseflow is at most half of observed flow. Summer storm volume."""
    return (
        pf.filter_by_month(SEASONS["summer"])
        .filter_by_scalar(BASEFLOW_FRACTION, le=0.50)
    )


# --- Daily view, flow duration: distribution only, timing ignored ---

def fdc(pf: PairedFrame) -> PairedFrame:
    """Daily means ranked on each side and paired by exceedance probability."""
    return pf.exceedance()


# --- Samples view: grab samples against the simulated value at the sample hour ---

def samples(pf: PairedFrame) -> PairedFrame:
    """Every sample."""
    return pf


# --- Partitions: which values are compared together ---

def _whole(pf: PairedFrame) -> PairedFrame:
    return pf


def _fdc_segments(pf: PairedFrame) -> PairedFrame:
    return pf.partition_by_exceedance_bins(FDC_SEGMENT_EDGES, FDC_SEGMENT_LABELS)


def _flow_regimes(pf: PairedFrame) -> PairedFrame:
    return pf.partition_by_bins(FLOW_EXCEEDANCE, FLOW_REGIME_EDGES, FLOW_REGIME_LABELS)


@dataclass(frozen=True)
class Partition:
    """Splits a transform's output into labelled groups."""

    key: str
    fn: Callable[[PairedFrame], PairedFrame]   # adds the label column ("none" leaves the frame whole)
    description: str
    needs_columns: tuple[str, ...] = ()

    def applies_to(self, pf: PairedFrame) -> bool:
        return all(c in pf.data.columns for c in self.needs_columns)

    def split(self, pf: PairedFrame) -> list[tuple[str, PairedFrame]]:
        return self.fn(pf).groups()


PARTITIONS: dict[str, Partition] = {p.key: p for p in [
    Partition("none", _whole, "All values together, labelled 'all'."),
    Partition("month", PairedFrame.partition_by_month, "Calendar month, pooled across years (jan-dec)."),
    Partition("season", PairedFrame.partition_by_season, "Season, pooled across years (winter-fall)."),
    Partition("year", PairedFrame.partition_by_year, "Calendar year; only years with data."),
    Partition("year_month", PairedFrame.partition_by_year_month, "Month window (YYYY-MM); only windows with data."),
    Partition("fdc_segment", _fdc_segments, "HSPEXP flow duration segments: high_10, mid_40, low_50."),
    Partition("flow_regime", _flow_regimes, "EPA load duration curve zones from flow_exceedance.",
              needs_columns=(FLOW_EXCEEDANCE,)),
]}


@dataclass(frozen=True)
class Grouping:
    """How one partition of a transform's output is compared."""

    comparisons: tuple[str, ...]   # comparisons that suit these groups
    min_points: int = 1            # recommended points per group, in the transform's own unit


@dataclass(frozen=True)
class ComparisonResult:
    """One comparison on one group of a transform's output."""

    transform: str
    by: str
    label: str
    comparison: str
    value: float
    status: str                                # "ok" | "undefined" (arithmetic failed) | "empty" (no points)
    units: str
    perfect: float
    n_points: int                              # points compared, after min_coverage
    required_points: int                       # max of the grouping's and the comparison's recommendation
    sufficient: bool                           # n_points >= required_points
    n_excluded: int = 0                        # points left out by min_coverage
    observed_value: Optional[float] = None     # statistic of each side, for <error>_of_<statistic>
    simulated_value: Optional[float] = None
    observed_mean: float = np.nan              # mean of each side over the group
    simulated_mean: float = np.nan
    coverage_min: float = np.nan               # share of base timesteps behind the points; NaN if not tracked
    coverage_mean: float = np.nan


def _coverage(pf: PairedFrame) -> tuple[float, float]:
    """(min, mean) coverage of a frame's points; 1.0 at the base timestep, NaN if not tracked."""
    if pf.n_points == 0:
        return np.nan, np.nan
    if COVERAGE in pf.data.columns:
        coverage = pf.data[COVERAGE]
        return float(coverage.min()), float(coverage.mean())
    if pf.view == "samples" or pf.index_kind != "time":
        return np.nan, np.nan
    return 1.0, 1.0


@dataclass(frozen=True)
class Transform:
    key: str
    fn: Callable[[PairedFrame], PairedFrame]
    view: str                              # "hourly" | "daily" | "samples"
    by: Mapping[str, Grouping]             # partition key -> comparisons and points per group
    needs_columns: tuple[str, ...] = ()    # columns the chain reads besides observed/simulated
    needs_extensive: bool = False          # totals: only for quantities that can be summed

    def __post_init__(self):
        for by_key, grouping in self.by.items():
            if by_key not in PARTITIONS:
                raise ConfigError(f"Transform '{self.key}': unknown partition '{by_key}'")
            unknown = [c for c in grouping.comparisons if c not in COMPARISONS]
            if unknown:
                raise ConfigError(f"Transform '{self.key}' by '{by_key}': unknown comparisons {unknown}")

    @property
    def description(self) -> str:
        return inspect.getdoc(self.fn) or ""

    def applies_to(self, pf: PairedFrame) -> bool:
        """True when ``pf`` is this transform's view and has what the chain needs."""
        return (
            pf.view == self.view
            and all(c in pf.data.columns for c in self.needs_columns)
            and (pf.quantity_kind == "extensive" or not self.needs_extensive)
        )

    def apply(self, pf: PairedFrame) -> PairedFrame:
        """Run the chain and return the transformed frame."""
        if pf.view != self.view:
            raise ConfigError(f"Transform '{self.key}' needs a {self.view} frame; got a {pf.view} frame")
        return self.fn(pf)

    # TODO: compute single comparison value
    # def comparison(self, pf: PairedFrame, name: str, min_points = None) -> list[ComparisonResult]:
    #     if min_points is None:
    #         min_points = self.min_points

    #     if name not in self.comparisons:
    #         raise ConfigError(f"Comparison '{name}' is not valid for transform '{self.key}'")

    #     if n < self.min_points:
    #         return ComparisonResult(self.key, label, name, np.nan, n, "insufficient_points")
    #     o, s = group.observed, group.simulated
    #     value = float(COMPARISONS[name](o, s))
    #     statistic = MARGINAL_STATISTICS.get(name)
    #     results.append(ComparisonResult(
    #         self.key, label, name, value, n,
    #         "ok" if np.isfinite(value) else "undefined",
    #         statistic(o) if statistic else None,
    #         statistic(s) if statistic else None,
    #     ))
    #     return results
    
    def compute(
        self,
        pf: PairedFrame,
        by: str = "none",
        min_coverage: Optional[float] = None,
        min_points: Optional[int] = None,
    ) -> list[ComparisonResult]:
        """Run the transform, split it ``by`` a partition, and run every listed comparison on each group.

        Points below ``min_coverage`` are left out of each group first and
        counted as ``n_excluded``. Every comparison then returns a value when
        the arithmetic allows; ``sufficient`` records whether the group meets
        ``required_points``. ``min_points`` overrides the grouping's
        recommendation.
        """
        if by not in self.by:
            raise ConfigError(f"Transform '{self.key}' has no partition '{by}'; choose from {tuple(self.by)}")
        grouping = self.by[by]
        group_points = grouping.min_points if min_points is None else min_points
        results = []
        for label, group in PARTITIONS[by].split(self.apply(pf)):
            n_before = group.n_points
            group = group.filter_by_coverage(min_coverage)
            n = group.n_points
            o, s = group.observed, group.simulated
            coverage_min, coverage_mean = _coverage(group)
            observed_mean, simulated_mean = (float(o.mean()), float(s.mean())) if n else (np.nan, np.nan)
            for name in grouping.comparisons:
                comparison = COMPARISONS[name]
                if n == 0:
                    value, observed_value, simulated_value, status = np.nan, None, None, "empty"
                else:
                    value, observed_value, simulated_value = comparison.evaluate(o, s)
                    status = "ok" if np.isfinite(value) else "undefined"
                required = max(group_points, comparison.min_points)
                results.append(ComparisonResult(
                    transform=self.key, by=by, label=label, comparison=name,
                    value=value, status=status,
                    units=comparison.units_for(pf.units), perfect=comparison.perfect,
                    n_points=n, required_points=required, sufficient=n >= required,
                    n_excluded=n_before - n,
                    observed_value=observed_value, simulated_value=simulated_value,
                    observed_mean=observed_mean, simulated_mean=simulated_mean,
                    coverage_min=coverage_min, coverage_mean=coverage_mean,
                ))
        return results


TRANSFORMS: dict[str, Transform] = {t.key: t for t in [
    # hourly view
    Transform("hourly_mean", hourly_mean, "hourly", {
        "none": Grouping(ALL, MIN_HOURS),
        "year": Grouping(ALL, MIN_HOURS),
    }),
    Transform("annual_peak", annual_peak, "hourly", {
        "none": Grouping(ALL, MIN_YEARS),
        "year": Grouping(ONE_VALUE),
    }),
    Transform("high_flow_hours", high_flow_hours, "hourly", {
        "none": Grouping(ERRORS, MIN_HOURS),
    }),
    # daily view: time series
    Transform("daily_mean", daily_mean, "daily", {
        "none": Grouping(ALL, MIN_DAYS),
        "month": Grouping(ALL, MIN_GROUP_DAYS),
        "season": Grouping(ALL, MIN_GROUP_DAYS),
        "year": Grouping(ALL, MIN_YEAR_DAYS),
        "year_month": Grouping(ALL, MIN_MONTH_DAYS),
    }),
    Transform("monthly_mean", monthly_mean, "daily", {
        "none": Grouping(ALL, MIN_MONTHS),
        "month": Grouping(ALL, MIN_YEARS),
        "year_month": Grouping(ONE_VALUE),
    }),
    Transform("annual_mean", annual_mean, "daily", {
        "none": Grouping(ALL, MIN_YEARS),
        "year": Grouping(ONE_VALUE),
    }),
    Transform("monthly_total", monthly_total, "daily", {
        "none": Grouping(ALL, MIN_MONTHS),
        "month": Grouping(ALL, MIN_YEARS),
        "year_month": Grouping(ONE_VALUE),
    }, needs_extensive=True),
    Transform("annual_total", annual_total, "daily", {
        "none": Grouping(ALL, MIN_YEARS),
        "year": Grouping(ONE_VALUE),
    }, needs_extensive=True),
    Transform("annual_max", annual_daily_max, "daily", {
        "none": Grouping(ALL, MIN_YEARS),
        "year": Grouping(ONE_VALUE),
    }),
    # daily view: flow regime on observed dates
    Transform("high_flow_days", high_flow_days, "daily", {"none": Grouping(ERRORS, MIN_DAYS)}),
    Transform("low_flow_days", low_flow_days, "daily", {"none": Grouping(ERRORS, MIN_DAYS)}),
    Transform("stormflow_days", stormflow_days, "daily", {"none": Grouping(ALL, MIN_DAYS)},
              needs_columns=(BASEFLOW_FRACTION,)),
    Transform("baseflow_days", baseflow_days, "daily", {"none": Grouping(ALL, MIN_DAYS)},
              needs_columns=(BASEFLOW_FRACTION,)),
    Transform("spring_high_flow_days", spring_high_flow_days, "daily", {"none": Grouping(ERRORS, MIN_DAYS)}),
    Transform("summer_low_flow_days", summer_low_flow_days, "daily", {"none": Grouping(ERRORS, MIN_DAYS)}),
    Transform("summer_stormflow_days", summer_stormflow_days, "daily", {"none": Grouping(ALL, MIN_DAYS)},
              needs_columns=(BASEFLOW_FRACTION,)),
    # daily view: flow duration
    Transform("fdc", fdc, "daily", {
        "none": Grouping(POINTWISE, MIN_DAYS),
        "fdc_segment": Grouping(ERRORS, MIN_GROUP_DAYS),
    }),
    # samples view
    Transform("samples", samples, "samples", {
        "none": Grouping(ALL, MIN_SAMPLES),
        "month": Grouping(ERRORS, MIN_SAMPLES_PER_MONTH),
        "season": Grouping(ERRORS, MIN_SAMPLES_PER_GROUP),
        "year": Grouping(ERRORS, MIN_SAMPLES_PER_GROUP),
        "flow_regime": Grouping(ERRORS, MIN_SAMPLES_PER_GROUP),
    }),
]}


def views(pf: PairedFrame, min_day_coverage: Optional[float] = None) -> dict[str, PairedFrame]:
    """The views a frame provides, keyed by view name.

    An hourly frame provides itself and its daily view; with
    ``min_day_coverage`` set, days with less coverage are left out of the daily
    view (``None`` keeps every day). Daily and samples frames provide only
    themselves.
    """
    if pf.index_kind != "time" or pf.is_aggregated or pf.is_partitioned:
        raise ConfigError("views needs an untransformed frame at its base timestep")
    if pf.view == "hourly":
        return {"hourly": pf, "daily": pf.to_daily().filter_by_coverage(min_day_coverage)}
    if pf.view == "daily":
        return {"daily": pf.filter_by_coverage(min_day_coverage)}
    return {"samples": pf}


def compute_all(
    pf: PairedFrame,
    min_day_coverage: Optional[float] = None,
    min_coverage: Optional[float] = None,
    keys: Optional[Iterable[str]] = None,
) -> list[ComparisonResult]:
    """Run every transform (or only ``keys``) and each of its partitions on the matching view.

    Transforms whose view the frame doesn't provide, or that need a column or
    an extensive quantity the frame lacks, are skipped, as are partitions that
    need a column the frame lacks.
    """
    available = views(pf, min_day_coverage)
    selected = TRANSFORMS.values() if keys is None else [TRANSFORMS[k] for k in keys]
    results = []
    for transform in selected:
        frame = available.get(transform.view)
        if frame is None or not transform.applies_to(frame):
            continue
        for by in transform.by:
            if PARTITIONS[by].applies_to(frame):
                results.extend(transform.compute(frame, by, min_coverage))
    return results


def describe(pf: PairedFrame, by: str = "none") -> pd.DataFrame:
    """One row per group: label, n_points, each side's mean and coverage. No comparison involved.

    ``describe(TRANSFORMS["annual_mean"].apply(daily), "year")`` gives each
    year's observed and simulated average and how complete the year is.
    """
    rows = []
    for label, group in PARTITIONS[by].split(pf):
        n = group.n_points
        coverage_min, coverage_mean = _coverage(group)
        rows.append({
            "label": label,
            "n_points": n,
            "observed_mean": float(group.observed.mean()) if n else np.nan,
            "simulated_mean": float(group.simulated.mean()) if n else np.nan,
            "coverage_min": coverage_min,
            "coverage_mean": coverage_mean,
        })
    return pd.DataFrame(rows, columns=["label", "n_points", "observed_mean", "simulated_mean",
                                       "coverage_min", "coverage_mean"])
