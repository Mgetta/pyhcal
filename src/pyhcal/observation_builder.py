# -*- coding: utf-8 -*-
"""
Observation Builder Module for PEST++ Workflows

This module provides functions for building PEST observations from hydrological
data. It separates the observation-building logic from the main PestWorkflow class
to improve testability and reduce code complexity.

Functions are designed to be pure and composable - they take inputs and return
outputs without side effects (except for optional print statements).
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from pyhcal.metrics import (
    METRIC_MONTH_NAMES,
    METRIC_SEASONS,
)

# Use constants from metrics module
MONTH_NAMES = METRIC_MONTH_NAMES
SEASONS = METRIC_SEASONS

# Default target values for summary statistics
DEFAULT_TARGETS = {
    'nse': 0.7,
    'log_nse': 0.5,
    'kge': 0.7,
    'bias': 0.0,
    'pbias': 0.0
}


def align_timeseries(
    observed: pd.Series,
    simulated: pd.Series,
    verbose: bool = True
) -> Tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """
    Find overlapping dates between observed and simulated timeseries.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
    verbose : bool
        Print summary of alignment
        
    Returns
    -------
    obs_overlap : pd.Series
        Observed data for overlapping dates only
    sim_overlap : pd.Series
        Simulated data for overlapping dates only
    common_dates : pd.DatetimeIndex
        The overlapping dates
        
    Raises
    ------
    ValueError
        If no overlapping dates exist
    """
    obs_clean = observed.dropna()
    sim_clean = simulated.dropna()
    common_dates = obs_clean.index.intersection(sim_clean.index)
    
    if len(common_dates) == 0:
        raise ValueError(
            "No overlapping dates between observed and simulated data! "
            "Check that your date ranges align."
        )
    
    obs_overlap = obs_clean.loc[common_dates].sort_index()
    sim_overlap = sim_clean.loc[common_dates].sort_index()
    
    if verbose:
        print(f"Data overlap summary:")
        print(f"  Observed data range: {obs_clean.index.min()} to {obs_clean.index.max()}")
        print(f"  Simulated data range: {sim_clean.index.min()} to {sim_clean.index.max()}")
        print(f"  Overlapping dates: {len(common_dates)} values")
        print(f"  Date range used: {common_dates.min()} to {common_dates.max()}")
    
    return obs_overlap, sim_overlap, common_dates


def build_timeseries_observations(
    observed: pd.Series,
    prefix: str,
    weight: float = 1.0,
    group_suffix: str = 'ts'
) -> List[Dict]:
    """
    Build timeseries observations (1-to-1 comparisons).
    
    Each timestep with observed data becomes a separate PEST observation.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index (should be pre-aligned)
    prefix : str
        Prefix for observation names (e.g., 'st1' -> 'st1_20050101')
    weight : float
        Weight for these observations
    group_suffix : str
        Suffix for group name (default 'ts' for timeseries)
        
    Returns
    -------
    List[Dict]
        List of observation records with keys: obsnme, obsval, weight, obgnme
    """
    observations = []
    
    for date, obs_val in observed.items():
        date_str = date.strftime('%Y%m%d')
        obs_name = f"{prefix}_{date_str}"
        
        observations.append({
            'obsnme': obs_name[:20],  # PEST limit
            'obsval': float(obs_val),
            'weight': weight,
            'obgnme': f'{prefix}_{group_suffix}'
        })
    
    return observations


def build_statistic_observations(
    observed: pd.Series,
    prefix: str,
    statistics: List[str],
    weight: float = 10.0,
    target_values: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Build summary statistic observations.
    
    For statistics, the 'obsval' is the TARGET value we want the model to achieve.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index (should be pre-aligned)
    prefix : str
        Prefix for observation names
    statistics : List[str]
        List of statistics to include. Options:
        'nse', 'log_nse', 'kge', 'bias', 'pbias', 
        'monthly_avg', 'seasonal', 'annual_avg'
    weight : float
        Base weight for statistic observations
    target_values : Dict[str, float], optional
        Target values for statistics (e.g., {'nse': 0.7})
        
    Returns
    -------
    List[Dict]
        List of observation records
    """
    targets = DEFAULT_TARGETS.copy()
    if target_values is not None:
        targets.update(target_values)
    
    observations = []
    
    for stat in statistics:
        stat_lower = stat.lower()
        
        if stat_lower == 'nse':
            observations.append({
                'obsnme': f'{prefix}_nse',
                'obsval': targets['nse'],
                'weight': weight,
                'obgnme': f'{prefix}_stat'
            })
            
        elif stat_lower == 'log_nse':
            observations.append({
                'obsnme': f'{prefix}_lnse',
                'obsval': targets['log_nse'],
                'weight': weight,
                'obgnme': f'{prefix}_stat'
            })
            
        elif stat_lower == 'kge':
            observations.append({
                'obsnme': f'{prefix}_kge',
                'obsval': targets['kge'],
                'weight': weight,
                'obgnme': f'{prefix}_stat'
            })
            
        elif stat_lower == 'bias':
            observations.append({
                'obsnme': f'{prefix}_bias',
                'obsval': targets['bias'],
                'weight': weight,
                'obgnme': f'{prefix}_stat'
            })
            
        elif stat_lower == 'pbias':
            observations.append({
                'obsnme': f'{prefix}_pbias',
                'obsval': targets['pbias'],
                'weight': weight / 100.0,  # Scale for percentage
                'obgnme': f'{prefix}_stat'
            })
            
        elif stat_lower == 'monthly_avg':
            observations.extend(
                _build_monthly_observations(observed, prefix, weight)
            )
            
        elif stat_lower == 'seasonal':
            observations.extend(
                _build_seasonal_observations(observed, prefix, weight)
            )
            
        elif stat_lower == 'annual_avg':
            observations.append({
                'obsnme': f'{prefix}_ann',
                'obsval': float(observed.mean()),
                'weight': weight,
                'obgnme': f'{prefix}_stat'
            })
    
    return observations


def _build_monthly_observations(
    observed: pd.Series,
    prefix: str,
    weight: float
) -> List[Dict]:
    """Build monthly average observations."""
    observations = []
    monthly_obs = observed.groupby(observed.index.month).mean()
    
    for month, obs_val in monthly_obs.items():
        observations.append({
            'obsnme': f'{prefix}_{MONTH_NAMES[month-1]}',
            'obsval': float(obs_val),
            'weight': weight / 12.0,
            'obgnme': f'{prefix}_mon'
        })
    
    return observations


def _build_seasonal_observations(
    observed: pd.Series,
    prefix: str,
    weight: float
) -> List[Dict]:
    """Build seasonal average observations."""
    observations = []
    
    for season_name, months in SEASONS.items():
        mask = observed.index.month.isin(months)
        if mask.any():
            obs_val = observed.loc[mask].mean()
            observations.append({
                'obsnme': f'{prefix}_{season_name[:3]}',
                'obsval': float(obs_val),
                'weight': weight / 4.0,
                'obgnme': f'{prefix}_seas'
            })
    
    return observations


def build_exceedance_observations(
    observed: pd.Series,
    prefix: str,
    probabilities: List[float],
    weight: float = 5.0
) -> List[Dict]:
    """
    Build exceedance probability (flow duration curve) observations.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index (should be pre-aligned)
    prefix : str
        Prefix for observation names
    probabilities : List[float]
        Exceedance probabilities (e.g., [10, 25, 50, 75, 90])
    weight : float
        Weight for these observations
        
    Returns
    -------
    List[Dict]
        List of observation records
    """
    observations = []
    obs_sorted = np.sort(observed.values)[::-1]  # Descending
    n = len(obs_sorted)
    
    for prob in probabilities:
        idx = int(np.ceil(prob / 100.0 * n)) - 1
        idx = max(0, min(idx, n - 1))
        obs_val = obs_sorted[idx]
        
        observations.append({
            'obsnme': f'{prefix}_q{int(prob):02d}',
            'obsval': float(obs_val),
            'weight': weight,
            'obgnme': f'{prefix}_fdc'
        })
    
    return observations


def build_observations(
    observed: pd.Series,
    simulated: pd.Series,
    prefix: str = 'obs',
    include_timeseries: bool = True,
    include_statistics: bool = True,
    include_exceedance: bool = False,
    statistics: Optional[List[str]] = None,
    exceedance_probs: Optional[List[float]] = None,
    timeseries_weight: float = 1.0,
    statistics_weight: float = 10.0,
    exceedance_weight: float = 5.0,
    target_values: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build complete set of PEST observations from observed and simulated data.
    
    This is the main entry point that composes all the individual builders.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
    prefix : str
        Prefix for observation names
    include_timeseries : bool
        Include 1-to-1 timeseries observations
    include_statistics : bool
        Include summary statistic observations
    include_exceedance : bool
        Include flow duration curve observations
    statistics : List[str], optional
        Statistics to include (default: ['nse', 'bias', 'monthly_avg'])
    exceedance_probs : List[float], optional
        Exceedance probabilities (default: [10, 25, 50, 75, 90])
    timeseries_weight : float
        Weight for timeseries observations
    statistics_weight : float
        Weight for statistic observations
    exceedance_weight : float
        Weight for exceedance observations
    target_values : Dict[str, float], optional
        Target values for statistics
    verbose : bool
        Print summary information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: obsnme, obsval, weight, obgnme
    """
    if statistics is None:
        statistics = ['nse', 'bias', 'monthly_avg']
    if exceedance_probs is None:
        exceedance_probs = [10, 25, 50, 75, 90]
    
    # Step 1: Align timeseries
    obs_overlap, sim_overlap, _ = align_timeseries(
        observed, simulated, verbose=verbose
    )
    
    observations = []
    
    # Step 2: Build timeseries observations
    if include_timeseries:
        ts_obs = build_timeseries_observations(
            obs_overlap, prefix, timeseries_weight
        )
        observations.extend(ts_obs)
    
    # Step 3: Build statistic observations
    if include_statistics:
        stat_obs = build_statistic_observations(
            obs_overlap, prefix, statistics, 
            statistics_weight, target_values
        )
        observations.extend(stat_obs)
    
    # Step 4: Build exceedance observations
    if include_exceedance:
        exc_obs = build_exceedance_observations(
            obs_overlap, prefix, exceedance_probs, exceedance_weight
        )
        observations.extend(exc_obs)
    
    # Create DataFrame
    obs_df = pd.DataFrame(observations)
    
    if verbose:
        _print_observation_summary(observations)
    
    return obs_df


def _print_observation_summary(observations: List[Dict]) -> None:
    """Print summary of created observations."""
    ts_count = len([o for o in observations if '_ts' in o['obgnme']])
    stat_count = len([o for o in observations 
                      if any(s in o['obgnme'] for s in ['_stat', '_mon', '_seas'])])
    exc_count = len([o for o in observations if '_fdc' in o['obgnme']])
    
    print(f"\nObservations created:")
    print(f"  Timeseries: {ts_count}")
    print(f"  Statistics: {stat_count}")
    print(f"  Exceedance: {exc_count}")
    print(f"  Total: {len(observations)}")


class ObservationBuilder:
    """
    Class-based interface for building observations.
    
    Provides a fluent interface for building observations step by step,
    with the ability to add multiple locations/constituents.
    
    Examples
    --------
    >>> builder = ObservationBuilder()
    >>> builder.add_location('st1', obs_series1, sim_series1, weight=1.0)
    >>> builder.add_location('st2', obs_series2, sim_series2, weight=2.0)
    >>> obs_df = builder.build()
    """
    
    def __init__(self):
        """Initialize empty builder."""
        self._locations = {}
        self._settings = {
            'include_timeseries': True,
            'include_statistics': True,
            'include_exceedance': False,
            'statistics': ['nse', 'bias', 'monthly_avg'],
            'exceedance_probs': [10, 25, 50, 75, 90],
            'target_values': None
        }
    
    def configure(
        self,
        include_timeseries: bool = True,
        include_statistics: bool = True,
        include_exceedance: bool = False,
        statistics: Optional[List[str]] = None,
        exceedance_probs: Optional[List[float]] = None,
        target_values: Optional[Dict[str, float]] = None
    ) -> 'ObservationBuilder':
        """Configure global settings for all locations."""
        self._settings['include_timeseries'] = include_timeseries
        self._settings['include_statistics'] = include_statistics
        self._settings['include_exceedance'] = include_exceedance
        if statistics is not None:
            self._settings['statistics'] = statistics
        if exceedance_probs is not None:
            self._settings['exceedance_probs'] = exceedance_probs
        if target_values is not None:
            self._settings['target_values'] = target_values
        return self
    
    def add_location(
        self,
        name: str,
        observed: pd.Series,
        simulated: pd.Series,
        weight: float = 1.0,
        group: Optional[str] = None
    ) -> 'ObservationBuilder':
        """
        Add a monitoring location.
        
        Parameters
        ----------
        name : str
            Short name for this location (used as prefix)
        observed : pd.Series
            Observed data with datetime index
        simulated : pd.Series
            Simulated data with datetime index
        weight : float
            Base weight multiplier for this location
        group : str, optional
            Custom group name for PEST obgnme
        """
        self._locations[name] = {
            'observed': observed,
            'simulated': simulated,
            'weight': weight,
            'group': group or name
        }
        return self
    
    def build(self, verbose: bool = True) -> pd.DataFrame:
        """
        Build observations for all locations.
        
        Returns
        -------
        pd.DataFrame
            Combined observation data
        """
        if not self._locations:
            raise ValueError("No locations added. Use add_location() first.")
        
        all_obs = []
        
        for name, loc_data in self._locations.items():
            obs_df = build_observations(
                observed=loc_data['observed'],
                simulated=loc_data['simulated'],
                prefix=name,
                include_timeseries=self._settings['include_timeseries'],
                include_statistics=self._settings['include_statistics'],
                include_exceedance=self._settings['include_exceedance'],
                statistics=self._settings['statistics'],
                exceedance_probs=self._settings['exceedance_probs'],
                timeseries_weight=loc_data['weight'],
                statistics_weight=loc_data['weight'] * 10.0,
                exceedance_weight=loc_data['weight'] * 5.0,
                target_values=self._settings['target_values'],
                verbose=verbose
            )
            
            # Replace group prefix if custom group specified
            if loc_data['group'] != name:
                obs_df['obgnme'] = obs_df['obgnme'].str.replace(
                    f'^{name}_', f"{loc_data['group']}_", regex=True
                )
            
            all_obs.append(obs_df)
        
        combined = pd.concat(all_obs, ignore_index=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"COMBINED: {len(combined)} observations from {len(self._locations)} locations")
            print(f"{'='*60}")
        
        return combined
