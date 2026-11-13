"""Observed/simulated evaluation: source objects in, ordinary DataFrames out.

No files, clocks, hashes, or model state are managed here. Callers supply audit
metadata; sources supply already-converted quantities. Model-internal constraint
evaluation is deferred until the reports-to-constraints contract is designed.
"""

from dataclasses import fields
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime

from pyhcal import metrics
from pyhcal.config.records import CalibrationConfig, Location, Period
from pyhcal.evaluation import registry
from pyhcal.eval.schema import Comparison, EvaluationResult, to_dataframe
from pyhcal.evaluation.transforms import get_transform


def build_comparison(
    run_id: str,
    cal,
    location: Location,
    constituent: str,
    timestep: str,
    units: str,
) -> pd.DataFrame:

    station_ids = [station_ref[0] for station_ref in location.station_refs]
    flow_station_ids = [flow_station_ref[0] for flow_station_ref in location.flow_station_refs]
    reach_ids = [reach_term[0] for reach_term in location.reach_terms]


    df = cal.compare_simulated_observed(station_ids,reach_ids,constituent,timestep, flow_station_ids=flow_station_ids)

    pf = PairedFrame(df,location.location_id,
                     constituent,
                     timestep,
                     units = units,
                     quantity_kind = quantity_kind)
    
    df['location_id'] = location.location_id
    df['constituent'] = constituent
    df['timestep'] = timestep
    df['unit'] = units
    df['run_id'] = run_id

    return df


def evaluate(run_id,cal, config):

    rows = []
    for eval_spec in config.eval_specs:

        location = [location for location in config.locations if location.location_id == eval_spec.location_id][0]

        station_ids = [station_ref[0] for station_ref in location.station_refs]
        reach_ids = [reach_term[0] for reach_term in location.reach_terms]
        window = [period for period in config.periods if period.period == eval_spec.period][0]
        transform = get_transform(eval_spec.transform)
        statistic = registry.get_statistic(eval_spec.statistic)

        obs = cal.get_observed_data(station_ids, eval_spec.constituent, eval_spec.timestep)
        obs = obs.loc[window.start_date:window.end_date]
        sim = cal.get_simulated_output(reach_ids, eval_spec.constituent, eval_spec.timestep)[0] #TODO: return a series inside of these functions. Don't use a magic number...
        sim = sim.loc[window.start_date:window.end_date]

        obs, sim = metrics.align_series(obs, sim)

        obs = transform(obs)
        sim = transform(sim)


        result = statistic.function(obs,sim)

        row = EvaluationResult(
            run_id,
            config_id = None,
            location_id=location.location_id,
            constituent=eval_spec.constituent,
            period=eval_spec.period,
            timestep=eval_spec.timestep,
            input_units=eval_spec.units,
            transform=eval_spec.transform,
            statistic=eval_spec.statistic,
            n_obs = len(obs),
            period_start = window.start_date,
            period_end = window.end_date,
            units=statistic.units,
            evaluated_at= datetime.now(),
            engine_version = 'n/a',
            label = statistic.display_name,
            value=result,
            target_min=eval_spec.target_min,
            target_max=eval_spec.target_max,
        )
        rows.append(row)

    
    return pd.DataFrame(rows)

    



