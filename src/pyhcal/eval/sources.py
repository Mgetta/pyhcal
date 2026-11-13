"""Minimal data-source contracts used by the comparison engine.

Adapters own extraction, station pooling, unit conversion, and physically valid
reach combinations. Returned Series must have unique, naive model-local
DatetimeIndex labels at the requested input resolution. Missing data use NaN.
"""

from typing import Protocol, Tuple

import pandas as pd

from pyhcal.config.records import ReachTerm, StationRef


class ObservedSource(Protocol):
    """Supply pooled observations in the explicitly requested units."""

    def get_series(
        self,
        stations: Tuple[StationRef, ...],
        constituent: str,
        timestep: str,
        *,
        units: str,
    ) -> pd.Series:
        ...


class SimulatedSource(Protocol):
    """Supply an output location's simulated series in requested units.

    Coefficients describe additive quantities. Adapters must reject physically
    unsupported combinations rather than sum concentrations or temperatures.
    """

    def get_reach_series(
        self,
        run_id: str,
        reach_terms: Tuple[ReachTerm, ...],
        constituent: str,
        timestep: str,
        *,
        units: str,
    ) -> pd.Series:
        ...
