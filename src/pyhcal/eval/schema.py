"""Flat row records for evaluation tables and a generic DataFrame converter.

The dataclasses describe the shape and meaning of individual rows. They do not
implement table querying, persistence, or runtime validation; pandas owns those
operations once records have been converted to a DataFrame.
"""

from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from typing import Iterable, Optional, Type, TypeVar

import pandas as pd


T = TypeVar("T")


def to_dataframe(records: Iterable[T], *, record_type: Type[T]) -> pd.DataFrame:
    """Convert homogeneous dataclass records to a DataFrame.

    ``record_type`` is required so empty inputs retain the dataclass's columns.
    Values and dtypes are inferred by pandas; this helper does not validate field
    annotations, relationships, uniqueness, units, or nullability.
    """
    if not isinstance(record_type, type) or not is_dataclass(record_type):
        raise TypeError("record_type must be a dataclass type")

    columns = [field.name for field in fields(record_type)]
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, record_type):
            raise TypeError(
                "Record at position {} must be {}, not {}".format(
                    index, record_type.__name__, type(record).__name__,
                )
            )
        rows.append({column: getattr(record, column) for column in columns})

    return pd.DataFrame.from_records(rows, columns=columns)


@dataclass(frozen=True)
class Comparison:
    """One timestamp of observed and simulated data for a location/constituent.

    Main ``units`` describes observed/simulated values only. ``flow_units``
    describes all auxiliary flow and baseflow values when requested.
    """

    run_id: str
    config_hash: str
    location_id: str
    constituent: str
    timestep: str
    datetime: datetime
    units: str
    observed: Optional[float] = None
    simulated: Optional[float] = None
    observed_flow: Optional[float] = None
    simulated_flow: Optional[float] = None
    observed_baseflow: Optional[float] = None
    simulated_baseflow: Optional[float] = None
    n_obs_pooled: Optional[int] = None
    flow_units: Optional[str] = None


@dataclass(frozen=True)
class EvaluationResult:
    """One evaluated statistic for a run, location, constituent, and period."""

    run_id: str
    location_id: str
    constituent: str
    period: str
    timestep: str
    input_units: str
    transform: str
    statistic: str
    n_obs: int
    period_start: date
    period_end: date
    units: str
    config_id: str
    evaluated_at: datetime
    engine_version: str
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    weight: Optional[float] = None
    label: Optional[str] = None
    value: Optional[float] = None
    obs_group: Optional[str] = None
    obs_value: Optional[float] = None
    sim_value: Optional[float] = None
    n_obs: Optional[int] = None
    n_raw: Optional[int] = None
    


@dataclass(frozen=True)
class ConstraintResult:
    """One model-output constraint evaluation, independent of station data."""

    run_id: str
    scope_type: str
    constituent: str
    statistic: str
    period: str
    units: str
    period_start: date
    period_end: date
    config_hash: str
    evaluated_at: datetime
    engine_version: str
    scope_id: Optional[str] = None
    value: Optional[float] = None
    target: Optional[float] = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    meets_target: Optional[bool] = None


@dataclass(frozen=True)
class RunRecord:
    """One simulation's identity and provenance."""

    run_id: str
    model_name: str
    run_name: str
    created_at: datetime
    uci_hash: str
    source: str
    sim_start: date
    sim_end: date
    parent_run_id: Optional[str] = None
    uci_path: Optional[str] = None
    tags: Optional[str] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class LocationRecord:
    """One flattened location metadata row for reporting and spatial joins."""

    config_id: str
    location_id: str
    model_name: str
    location_name: str
    kind: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    drainage_area_sqmi: Optional[float] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class LocationReach:
    """One reach membership/coefficient row for a configured location."""

    config_id: str
    location_id: str
    reach_id: int
    coefficient: float
    role: str


@dataclass(frozen=True)
class LocationStation:
    """One station membership and use at a configured location."""

    config_id: str
    location_id: str
    station_id: str
    station_origin: str
    role: str


@dataclass(frozen=True)
class ConstituentRecord:
    """One hub constituent's descriptive metadata."""

    constituent_id: str
    display_name: str
    kind: str
    agg_method: str
    is_derived: bool
    units_conc: Optional[str] = None
    units_load: Optional[str] = None
    derived_from: Optional[str] = None
