"""Comparison functions: each maps paired observed/simulated arrays to one number.

Functions return NaN only where the arithmetic breaks down: too few values for
the calculation (1 for mean-type statistics and percentiles, 2 for spread and
nse, 3 for correlations) or a zero denominator.

Each function is registered as a ``Comparison`` that records its kind, units,
perfect value and the number of values it needs for a meaningful result
(``min_points``). That number is reported with each result, not enforced.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.stats


MIN_SPREAD_VALUES = 2         # std, cv and nse need variation between values
MIN_CORRELATION_PAIRS = 3     # two points always correlate perfectly


def _divide(numerator, denominator):
    """Divide, giving NaN where the denominator is zero."""
    denominator = np.where(np.asarray(denominator) == 0, np.nan, denominator)
    return numerator / denominator


def _pearson(observed, simulated) -> float:
    if len(observed) < MIN_CORRELATION_PAIRS or np.std(observed) == 0 or np.std(simulated) == 0:
        return np.nan
    return float(np.corrcoef(observed, simulated)[0, 1])


def _spearman(observed, simulated) -> float:
    if len(observed) < MIN_CORRELATION_PAIRS or np.ptp(observed) == 0 or np.ptp(simulated) == 0:
        return np.nan
    return float(scipy.stats.spearmanr(observed, simulated).statistic)


# univariate Statistical functions for evaluating arrays of values.
# Each returns NaN when it has fewer values than the calculation needs.

def mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) >= 1 else np.nan

def median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) >= 1 else np.nan

def sum(values: np.ndarray) -> float:
    return float(np.sum(values)) if len(values) >= 1 else np.nan

def min(values: np.ndarray) -> float:
    return float(np.min(values)) if len(values) >= 1 else np.nan

def max(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) >= 1 else np.nan

def std(values: np.ndarray) -> float:
    return float(np.std(values)) if len(values) >= MIN_SPREAD_VALUES else np.nan

def cv(values: np.ndarray) -> float:
    return float(_divide(np.std(values), np.mean(values))) if len(values) >= MIN_SPREAD_VALUES else np.nan

def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if len(values) >= 1 else np.nan

def count(values: np.ndarray) -> float:
    return float(len(values))

def percentile(values: np.ndarray, p: int):
    return float(np.percentile(values, p)) if len(values) >= 1 else np.nan

def p10(values: np.ndarray) -> float:
    return percentile(values, 10)

def p25(values: np.ndarray) -> float:
    return percentile(values, 25)

def p75(values: np.ndarray) -> float:
    return percentile(values, 75)

def p90(values: np.ndarray) -> float:
    return percentile(values, 90)

# Error functions. Percent and ratio errors are NaN where observed is zero.

def error(observed: float, simulated: float):
    return simulated - observed

def abs_error(observed: float, simulated: float):
    return abs(simulated - observed)

def sq_error(observed: float, simulated: float):
    return (simulated - observed) ** 2

def pct_error(observed: float, simulated: float):
    return 100.0 * _divide(simulated - observed, observed)

def abs_pct_error(observed: float, simulated: float):
    return abs(pct_error(observed, simulated))

def ratio(observed: float, simulated: float):
    return _divide(simulated, observed)

def log_ratio(observed: float, simulated: float):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(ratio(observed, simulated))



# Score functions: joint functions of the paired values.
def rmse(observed: float, simulated: float):
    return rms(error(observed, simulated))

def nse(observed: float, simulated: float):
    if len(observed) < MIN_SPREAD_VALUES:
        return np.nan
    return 1 - _divide(np.sum((simulated - observed) ** 2), np.sum((observed - np.mean(observed)) ** 2))

def kge(observed: float, simulated: float):
    r = _pearson(observed, simulated)
    if np.isnan(r):
        return np.nan
    alpha = _divide(np.std(simulated), np.std(observed))
    beta = _divide(np.mean(simulated), np.mean(observed))
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def r(observed: float, simulated: float):
    return _pearson(observed, simulated)

def r2(observed: float, simulated: float):
    return _pearson(observed, simulated) ** 2

def spearman_r(observed: float, simulated: float):
    return _spearman(observed, simulated)

def spearman_r2(observed: float, simulated: float):
    return _spearman(observed, simulated) ** 2
    


# ---------------------------------------------------------------------------
# Named comparisons
#
# Each function takes paired observed/simulated arrays and returns one number.
# The name states the computation:
#
#   <score>                  nse, kge        joint function of the paired values
#   <error>_of_<statistic>   pct_error_of_mean   statistic on each side, then the
#                                                error between the two results
#   <statistic>_<error>      median_pct_error    error at each pair, then a
#                                                statistic over those errors
#
# Pairs where a percent or ratio error is undefined (observed == 0) are
# dropped before the statistic is taken.
# ---------------------------------------------------------------------------

# <error>_of_<statistic>

def error_of_mean(observed, simulated):
    return error(mean(observed), mean(simulated))

def error_of_median(observed, simulated):
    return error(median(observed), median(simulated))

def error_of_sum(observed, simulated):
    return error(sum(observed), sum(simulated))

def pct_error_of_mean(observed, simulated):
    return pct_error(mean(observed), mean(simulated))

def pct_error_of_median(observed, simulated):
    return pct_error(median(observed), median(simulated))

def pct_error_of_sum(observed, simulated):
    return pct_error(sum(observed), sum(simulated))

def pct_error_of_min(observed, simulated):
    return pct_error(min(observed), min(simulated))

def pct_error_of_max(observed, simulated):
    return pct_error(max(observed), max(simulated))

def pct_error_of_std(observed, simulated):
    return pct_error(std(observed), std(simulated))

def pct_error_of_p10(observed, simulated):
    return pct_error(p10(observed), p10(simulated))

def pct_error_of_p25(observed, simulated):
    return pct_error(p25(observed), p25(simulated))

def pct_error_of_p75(observed, simulated):
    return pct_error(p75(observed), p75(simulated))

def pct_error_of_p90(observed, simulated):
    return pct_error(p90(observed), p90(simulated))

def ratio_of_mean(observed, simulated):
    return ratio(mean(observed), mean(simulated))

def ratio_of_std(observed, simulated):
    return ratio(std(observed), std(simulated))

def ratio_of_cv(observed, simulated):
    return ratio(cv(observed), cv(simulated))


# <statistic>_<error>

def _defined(errors):
    return errors[np.isfinite(errors)]

def mean_error(observed, simulated):
    return mean(error(observed, simulated))

def mean_abs_error(observed, simulated):
    return mean(abs_error(observed, simulated))

def max_abs_error(observed, simulated):
    return max(abs_error(observed, simulated))

def mean_sq_error(observed, simulated):
    return mean(sq_error(observed, simulated))

def rms_error(observed, simulated):
    return rms(error(observed, simulated))

def mean_pct_error(observed, simulated):
    return mean(_defined(pct_error(observed, simulated)))

def median_pct_error(observed, simulated):
    return median(_defined(pct_error(observed, simulated)))

def mean_abs_pct_error(observed, simulated):
    return mean(_defined(abs_pct_error(observed, simulated)))

def median_abs_pct_error(observed, simulated):
    return median(_defined(abs_pct_error(observed, simulated)))

def max_abs_pct_error(observed, simulated):
    return max(_defined(abs_pct_error(observed, simulated)))

def p90_abs_pct_error(observed, simulated):
    return percentile(_defined(abs_pct_error(observed, simulated)), 90)

def mean_log_ratio(observed, simulated):
    return mean(_defined(log_ratio(observed, simulated)))


# Conventional names (each is exactly one of the functions above)

def pbias(observed, simulated):
    return pct_error_of_sum(observed, simulated)

def bias(observed, simulated):
    return error_of_mean(observed, simulated)

def mae(observed, simulated):
    return mean_abs_error(observed, simulated)

def mape(observed, simulated):
    return mean_abs_pct_error(observed, simulated)


@dataclass(frozen=True)
class Comparison:
    """One named comparison and what is known about it.

    ``min_points`` is the number of values the comparison needs for a
    meaningful result; it is reported with each result, never enforced here.
    ``statistic`` (marginal comparisons only) is applied to each side to report
    the two values being compared.
    """

    key: str
    fn: Callable[[np.ndarray, np.ndarray], float]   # the pure function
    kind: str                                       # "score" | "marginal" | "pointwise"
    units: str                                      # "%" | "-" | "native" | "native^2" (frame units)
    perfect: float                                  # 1 for nse, 0 for pbias, 1 for ratios
    min_points: int = 1                             # values needed for a meaningful result
    statistic: Optional[Callable[[np.ndarray], float]] = None   # marginal only: applied to each side

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Comparison '{self.key}': kind must be one of {KINDS}")
        if (self.kind == "marginal") != (self.statistic is not None):
            raise ValueError(f"Comparison '{self.key}': a statistic is needed exactly when kind is 'marginal'")

    def evaluate(self, observed: np.ndarray, simulated: np.ndarray) -> tuple[float, Optional[float], Optional[float]]:
        """(value, observed statistic, simulated statistic); the statistics are None unless marginal."""
        value = float(self.fn(observed, simulated))
        if self.statistic is None:
            return value, None, None
        return value, float(self.statistic(observed)), float(self.statistic(simulated))

    def units_for(self, frame_units: str) -> str:
        """Result units given the units of the values compared."""
        if self.units == "native":
            return frame_units
        if self.units == "native^2":
            return f"({frame_units})^2"
        return self.units


KINDS = ("score", "marginal", "pointwise")
MEANINGFUL_POINTS = 10        # scores and comparisons built on percentiles or spread

COMPARISONS: dict[str, Comparison] = {c.key: c for c in [
    # scores: joint functions of the paired values
    Comparison("nse", nse, "score", "-", 1.0, MEANINGFUL_POINTS),
    Comparison("kge", kge, "score", "-", 1.0, MEANINGFUL_POINTS),
    Comparison("r", r, "score", "-", 1.0, MEANINGFUL_POINTS),
    Comparison("r2", r2, "score", "-", 1.0, MEANINGFUL_POINTS),
    Comparison("spearman_r", spearman_r, "score", "-", 1.0, MEANINGFUL_POINTS),
    Comparison("spearman_r2", spearman_r2, "score", "-", 1.0, MEANINGFUL_POINTS),
    # <error>_of_<statistic>: statistic on each side, then the error between them
    Comparison("error_of_mean", error_of_mean, "marginal", "native", 0.0, statistic=mean),
    Comparison("error_of_median", error_of_median, "marginal", "native", 0.0, statistic=median),
    Comparison("error_of_sum", error_of_sum, "marginal", "native", 0.0, statistic=sum),
    Comparison("pct_error_of_mean", pct_error_of_mean, "marginal", "%", 0.0, statistic=mean),
    Comparison("pct_error_of_median", pct_error_of_median, "marginal", "%", 0.0, statistic=median),
    Comparison("pct_error_of_sum", pct_error_of_sum, "marginal", "%", 0.0, statistic=sum),
    Comparison("pct_error_of_min", pct_error_of_min, "marginal", "%", 0.0, statistic=min),
    Comparison("pct_error_of_max", pct_error_of_max, "marginal", "%", 0.0, statistic=max),
    Comparison("pct_error_of_std", pct_error_of_std, "marginal", "%", 0.0, MEANINGFUL_POINTS, std),
    Comparison("pct_error_of_p10", pct_error_of_p10, "marginal", "%", 0.0, MEANINGFUL_POINTS, p10),
    Comparison("pct_error_of_p25", pct_error_of_p25, "marginal", "%", 0.0, MEANINGFUL_POINTS, p25),
    Comparison("pct_error_of_p75", pct_error_of_p75, "marginal", "%", 0.0, MEANINGFUL_POINTS, p75),
    Comparison("pct_error_of_p90", pct_error_of_p90, "marginal", "%", 0.0, MEANINGFUL_POINTS, p90),
    Comparison("ratio_of_mean", ratio_of_mean, "marginal", "-", 1.0, statistic=mean),
    Comparison("ratio_of_std", ratio_of_std, "marginal", "-", 1.0, MEANINGFUL_POINTS, std),
    Comparison("ratio_of_cv", ratio_of_cv, "marginal", "-", 1.0, MEANINGFUL_POINTS, cv),
    Comparison("pbias", pbias, "marginal", "%", 0.0, statistic=sum),
    Comparison("bias", bias, "marginal", "native", 0.0, statistic=mean),
    # <statistic>_<error>: error at each pair, then a statistic over the errors
    Comparison("mean_error", mean_error, "pointwise", "native", 0.0),
    Comparison("mean_abs_error", mean_abs_error, "pointwise", "native", 0.0),
    Comparison("max_abs_error", max_abs_error, "pointwise", "native", 0.0),
    Comparison("mean_sq_error", mean_sq_error, "pointwise", "native^2", 0.0),
    Comparison("rms_error", rms_error, "pointwise", "native", 0.0),
    Comparison("rmse", rmse, "pointwise", "native", 0.0),
    Comparison("mean_pct_error", mean_pct_error, "pointwise", "%", 0.0),
    Comparison("median_pct_error", median_pct_error, "pointwise", "%", 0.0),
    Comparison("mean_abs_pct_error", mean_abs_pct_error, "pointwise", "%", 0.0),
    Comparison("median_abs_pct_error", median_abs_pct_error, "pointwise", "%", 0.0),
    Comparison("max_abs_pct_error", max_abs_pct_error, "pointwise", "%", 0.0),
    Comparison("p90_abs_pct_error", p90_abs_pct_error, "pointwise", "%", 0.0, MEANINGFUL_POINTS),
    Comparison("mean_log_ratio", mean_log_ratio, "pointwise", "-", 0.0),
    Comparison("mae", mae, "pointwise", "native", 0.0),
    Comparison("mape", mape, "pointwise", "%", 0.0),
]}

# Comparison names by kind, for deciding which suit a transform's output.
SCORES = tuple(k for k, c in COMPARISONS.items() if c.kind == "score")
MARGINALS = tuple(k for k, c in COMPARISONS.items() if c.kind == "marginal")
POINTWISE = tuple(k for k, c in COMPARISONS.items() if c.kind == "pointwise")
