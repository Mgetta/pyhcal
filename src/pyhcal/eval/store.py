"""Local evaluation mart initialization and basic Parquet I/O."""

import json
from dataclasses import fields
from datetime import date, datetime
from pathlib import Path
from typing import List, Type, Union, get_args, get_origin, get_type_hints

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    Comparison,
    ConstituentRecord,
    ConstraintResult,
    EvaluationResult,
    LocationReach,
    LocationRecord,
    LocationStation,
    PeriodRecord,
    RunRecord,
    StatisticRecord,
    TransformRecord,
)


_TABLE_RECORDS = {
    "runs": RunRecord,
    "locations": LocationRecord,
    "location_reaches": LocationReach,
    "location_stations": LocationStation,
    "constituents": ConstituentRecord,
    "metrics": StatisticRecord,
    "transforms": TransformRecord,
    "periods": PeriodRecord,
    "comparisons": Comparison,
    "results": EvaluationResult,
    "constraint_results": ConstraintResult,
}


def _arrow_schema(record_type: Type) -> pa.Schema:
    types = {
        str: pa.string(),
        int: pa.int64(),
        float: pa.float64(),
        bool: pa.bool_(),
        date: pa.date32(),
        datetime: pa.timestamp("us", tz="UTC"),
    }
    annotations = get_type_hints(record_type)
    columns = []
    for field in fields(record_type):
        annotation = annotations[field.name]
        nullable = False
        if get_origin(annotation) is Union:
            args = get_args(annotation)
            if len(args) != 2 or type(None) not in args:
                raise TypeError("Unsupported store annotation: {!r}".format(annotation))
            nullable = True
            annotation = next(arg for arg in args if arg is not type(None))
        if annotation not in types:
            raise TypeError("Unsupported store annotation: {!r}".format(annotation))
        # Comparison timestamps are model-local; audit timestamps are UTC.
        dtype = (
            pa.timestamp("us")
            if annotation is datetime and field.name == "datetime"
            else types[annotation]
        )
        columns.append(pa.field(field.name, dtype, nullable=nullable))
    return pa.schema(columns)


def create_store(
    store_path: Union[Path, str],
    *,
    engine_version: str,
    schema_version: str = "1.0",
) -> Path:
    """Create a zero-row evaluation mart and return its absolute root path.

    The supplied path is the mart itself, not its project parent. It may be
    missing or an empty directory; nonempty destinations are never overwritten.
    Every table gets a typed ``part-0.parquet`` derived from its row dataclass.
    Run/timestep partitions are created only when real data are written later.

    The manifest is written last. I/O errors propagate; initialization is not
    transactional and a failed write may leave a partial directory without a
    manifest. Do not call this concurrently for the same destination.
    """
    for name, version in (
        ("schema_version", schema_version), ("engine_version", engine_version),
    ):
        if not isinstance(version, str) or not version.strip():
            raise ValueError("{} must be a nonempty string".format(name))
    root = Path(store_path).resolve()
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError("Store path is not a directory: {}".format(root))
        if any(root.iterdir()):
            raise FileExistsError("Store directory is not empty: {}".format(root))

    schemas = {name: _arrow_schema(record) for name, record in _TABLE_RECORDS.items()}
    root.mkdir(parents=True, exist_ok=True)
    for name, schema in schemas.items():
        folder = root / name
        folder.mkdir()
        with (folder / "part-0.parquet").open("xb") as stream:
            pq.write_table(pa.Table.from_pylist([], schema=schema), stream)
    with (root / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"schema_version": schema_version, "engine_version": engine_version},
            stream,
            indent=2,
        )
        stream.write("\n")
    return root


def read_from_store(store_path: Union[Path, str]):
    store_path = Path(store_path)
    if store_path.exists():
        return pd.read_parquet(store_path)
    else:
        raise FileNotFoundError(f"Store path {store_path} does not exist.")

def write_to_store(store_path: Union[Path, str], rows: List):
    store_path = Path(store_path)
    #TODO: separate creating store from writing to store
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        df = pd.read_parquet(store_path)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.to_parquet(store_path, index=False)
