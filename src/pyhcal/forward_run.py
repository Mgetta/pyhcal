#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEST Forward Model Module

This module provides functions for running HSPF as a forward model within PEST++
parameter estimation. It handles model execution, data extraction, metric 
calculation, and output generation.

Functions can be used directly or via the generated forward_model.py script.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyhcal.metrics import calculate_all_statistics


@dataclass
class ForwardModelConfig:
    """Configuration for forward model execution."""
    model_exe: str = 'hspf'
    uci_file: str = 'model.uci'
    output_file: str = 'model_output.txt'
    observed_data_file: str = 'observed_data.csv'
    reach_map: Dict[str, int] = None
    observation_info: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.reach_map is None:
            self.reach_map = {}
        if self.observation_info is None:
            self.observation_info = {}


def run_hspf(model_exe: str, uci_file: str, working_dir: Optional[Path] = None) -> bool:
    """
    Run the HSPF model.
    
    Parameters
    ----------
    model_exe : str
        Path to HSPF executable
    uci_file : str
        Path to UCI file
    working_dir : Path, optional
        Working directory for model execution
        
    Returns
    -------
    bool
        True if model ran successfully, False otherwise
    """
    print(f"Running HSPF: {model_exe} {uci_file}")
    
    try:
        result = subprocess.run(
            [model_exe, uci_file], 
            capture_output=True, 
            text=True,
            cwd=working_dir
        )
        
        if result.returncode != 0:
            print(f"HSPF error (return code {result.returncode})")
            if result.stderr:
                print(f"stderr: {result.stderr}")
            if result.stdout:
                print(f"stdout: {result.stdout}")
            return False
            
        print("HSPF completed successfully")
        return True
        
    except FileNotFoundError:
        print(f"Error: Model executable not found: {model_exe}")
        return False
    except Exception as e:
        print(f"Error running HSPF: {e}")
        return False


def load_observed_data(
    filepath: Union[str, Path],
    date_column: str = 'date',
    value_column: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Load observed data from a CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    date_column : str
        Name of date column (default: 'date')
    value_column : str, optional
        Name of value column to extract. If None, returns all columns.
        
    Returns
    -------
    pd.DataFrame or None
        Observed data with datetime index, or None if file not found
    """
    if not os.path.exists(filepath):
        print(f"WARNING: Observed data file not found: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath, parse_dates=[date_column], index_col=date_column)
        
        if value_column and value_column in df.columns:
            return df[[value_column]]
        return df
        
    except Exception as e:
        print(f"Error loading observed data: {e}")
        return None


def extract_simulated_data(
    uci_file: str,
    reach_id: int,
    constituent: str = 'ROVOL',
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> Optional[pd.Series]:
    """
    Extract simulated data from HSPF output.
    
    This is a placeholder function - users should implement their own
    extraction logic based on their model setup.
    
    Parameters
    ----------
    uci_file : str
        Path to UCI file
    reach_id : int
        Reach ID to extract
    constituent : str
        Constituent name (default: 'ROVOL' for runoff volume)
    start_date : pd.Timestamp, optional
        Start date filter
    end_date : pd.Timestamp, optional
        End date filter
        
    Returns
    -------
    pd.Series or None
        Simulated timeseries with datetime index
        
    Notes
    -----
    CUSTOMIZE THIS FUNCTION FOR YOUR MODEL!
    
    Example implementation using pyhcal:
    
    >>> from hspf import UCI
    >>> uci = UCI(uci_file)
    >>> sim_data = uci.get_reach_output(reach_id, constituent)
    >>> return sim_data
    """
    print(f"WARNING: extract_simulated_data() not implemented!")
    print(f"  UCI file: {uci_file}")
    print(f"  Reach ID: {reach_id}")
    print(f"  Constituent: {constituent}")
    print("  Please customize this function for your model output format.")
    return None


def align_timeseries(
    observed: pd.Series, 
    simulated: pd.Series
) -> tuple:
    """
    Align observed and simulated timeseries to overlapping dates.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    tuple
        (aligned_observed, aligned_simulated) as pd.Series
    """
    common_index = observed.index.intersection(simulated.index)
    
    if len(common_index) == 0:
        print("WARNING: No overlapping dates between observed and simulated data")
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    return observed.loc[common_index], simulated.loc[common_index]


def calculate_observation_values(
    observed: pd.Series,
    simulated: pd.Series,
    obs_prefix: str,
    include_timeseries: bool = True,
    include_statistics: bool = True
) -> Dict[str, float]:
    """
    Calculate all observation values for PEST.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
    obs_prefix : str
        Prefix for observation names
    include_timeseries : bool
        Include 1-to-1 timeseries comparisons (default: True)
    include_statistics : bool
        Include summary statistics (default: True)
        
    Returns
    -------
    dict
        Dictionary of {observation_name: simulated_value}
    """
    results = {}
    
    # Align timeseries to overlapping dates
    obs_aligned, sim_aligned = align_timeseries(observed, simulated)
    
    if len(obs_aligned) == 0:
        return results
    
    # Calculate statistics using pyhcal.metrics
    if include_statistics:
        stats = calculate_all_statistics(obs_aligned, sim_aligned)
        for stat_name, value in stats.items():
            results[f'{obs_prefix}_{stat_name}'] = value
    
    # Add timeseries observations (simulated values at observed dates)
    if include_timeseries:
        for date, sim_val in sim_aligned.items():
            date_str = date.strftime('%Y%m%d')
            results[f'{obs_prefix}_{date_str}'] = sim_val
    
    return results


def write_pest_output(
    results: Dict[str, float],
    output_file: Union[str, Path],
    missing_value: float = -999999.0
) -> int:
    """
    Write observation values to PEST output file.
    
    Parameters
    ----------
    results : dict
        Dictionary of {observation_name: value}
    output_file : str or Path
        Path to output file
    missing_value : float
        Value to use for missing/NaN values (default: -999999.0)
        
    Returns
    -------
    int
        Number of values written
    """
    print(f"Writing output to {output_file}")
    
    count = 0
    with open(output_file, 'w') as f:
        for obs_name, value in sorted(results.items()):
            if value is not None and not np.isnan(value):
                f.write(f"{obs_name}  {value:.6f}\n")
            else:
                f.write(f"{obs_name}  {missing_value}\n")
            count += 1
    
    print(f"Wrote {count} values")
    return count


def run_forward_model(config: ForwardModelConfig) -> Dict[str, float]:
    """
    Run the complete forward model workflow.
    
    Parameters
    ----------
    config : ForwardModelConfig
        Configuration object
        
    Returns
    -------
    dict
        Dictionary of all observation values
        
    Notes
    -----
    This function:
    1. Runs HSPF model
    2. Loads observed data
    3. Extracts simulated data for each observation location
    4. Calculates statistics and timeseries values
    5. Writes output file for PEST
    """
    print("=" * 60)
    print("PEST Forward Model")
    print("=" * 60)
    
    # Step 1: Run HSPF
    success = run_hspf(config.model_exe, config.uci_file)
    if not success:
        print("ERROR: HSPF model failed")
        sys.exit(1)
    
    # Step 2: Load observed data
    observed_data = load_observed_data(config.observed_data_file)
    
    # Step 3: Extract simulated data and calculate metrics
    results = {}
    
    for obs_name, reach_id in config.reach_map.items():
        if reach_id is None:
            continue
            
        obs_info = config.observation_info.get(obs_name, {})
        constituent = obs_info.get('constituent', 'ROVOL')
        
        # Extract simulated data
        sim_data = extract_simulated_data(
            config.uci_file, 
            reach_id, 
            constituent
        )
        
        if sim_data is None:
            continue
        
        # Get observed data for this location
        if observed_data is not None:
            # Get appropriate column from observed data
            # Customize this based on your data structure
            obs_column = obs_info.get('obs_column', obs_name)
            if obs_column in observed_data.columns:
                obs_subset = observed_data[obs_column]
            else:
                obs_subset = observed_data.iloc[:, 0]  # First column as default
            
            # Calculate observations
            obs_results = calculate_observation_values(
                obs_subset, 
                sim_data, 
                obs_name,
                include_timeseries=True,
                include_statistics=True
            )
            results.update(obs_results)
    
    # Step 4: Write output file
    write_pest_output(results, config.output_file)
    
    print(f"Forward model complete. Generated {len(results)} observation values.")
    return results


def create_config_from_dict(
    model_exe: str,
    uci_file: str,
    output_file: str,
    observed_data_file: str,
    reach_map: Dict[str, int],
    observation_info: Dict[str, Dict[str, Any]]
) -> ForwardModelConfig:
    """
    Create a ForwardModelConfig from individual parameters.
    
    Convenience function for use in generated scripts.
    """
    return ForwardModelConfig(
        model_exe=model_exe,
        uci_file=uci_file,
        output_file=output_file,
        observed_data_file=observed_data_file,
        reach_map=reach_map,
        observation_info=observation_info
    )


# Entry point for command-line execution
if __name__ == '__main__':
    print("This module provides forward model functions for PEST++.")
    print("Use from pyhcal.forward_run import run_forward_model, ForwardModelConfig")
    print("Or use the generated forward_model.py script in your PEST directory.")
