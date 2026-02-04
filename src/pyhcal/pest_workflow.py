# -*- coding: utf-8 -*-
"""
PEST++ Workflow Module for HSPF Model Calibration

This module provides a workflow for setting up and running PEST++ parameter 
estimation and sensitivity analysis on HSPF hydrological models using pyemu.

PEST++ Background
-----------------
PEST++ is a software suite for parameter estimation, uncertainty quantification,
and sensitivity analysis. It implements various algorithms including:

- PESTPP-GLM: Gradient-based parameter estimation (similar to classic PEST)
- PESTPP-IES: Iterative ensemble smoother for history matching
- PESTPP-SEN: Global sensitivity analysis (Method of Morris, Sobol)
- PESTPP-OPT: Constrained optimization
- PESTPP-SWP: Parameter sweeping

HOW THE PEST++ OBJECTIVE FUNCTION WORKS
=======================================

This is the core concept to understand. PEST++ minimizes the weighted sum of 
squared residuals, called Phi (Φ):

    Φ = Σ [w_i × (observed_i - simulated_i)]²

Where:
- observed_i: The measured/target value (from observation data or calculated target)
- simulated_i: The model output (extracted from model output files)
- w_i: The weight for this observation

TYPES OF OBSERVATIONS
--------------------

**1. TIMESERIES OBSERVATIONS (1-to-1 comparisons)**

Each time step where you have both observed and simulated data becomes a 
separate observation in PEST:

    Date         Observed  Simulated  Residual
    1996-01-01   100       95         5
    1996-01-02   110       108        2
    1996-01-03   90        92        -2
    ...

Pros: Captures daily dynamics, many data points
Cons: Can be dominated by high flows, may weight wet periods heavily

**2. SUMMARY STATISTIC OBSERVATIONS (NSE, bias, monthly means, etc.)**

For statistics, the "observed" value is your TARGET, and the "simulated" 
value is what the model computes:

    Statistic    Target    Computed   Residual
    NSE          0.7       0.65       -0.05
    Bias         0.0       5.0         5.0
    Jan_avg      150       145        -5.0

Key insight: You're not comparing "observed NSE" vs "simulated NSE".
You're saying: "I want my model to achieve NSE ≥ 0.7" (the target).

**3. EXCEEDANCE PROBABILITY OBSERVATIONS (Flow Duration Curve)**

Compare simulated and observed flows at specific exceedance levels:

    Exceedance   Observed  Simulated  Residual
    Q10 (high)   500       480       -20
    Q50 (median) 100       105         5  
    Q90 (low)    20        18        -2

This captures whether the model gets high flows, low flows, and median right.

HANDLING DATA OVERLAP
--------------------

CRITICAL: Only use dates where BOTH observed AND simulated data exist!

    Simulated: |---------------------|  (full model run 1996-2020)
    Observed:        |------|           (limited: 2000-2010)
    Used:            |------|           (only the intersection!)

The `build_observations_from_data()` method handles this automatically.

WEIGHTING STRATEGIES
-------------------

The relative weights between observation types matter enormously!

Example: If you have 3650 daily observations and 1 NSE target:
- Equal total group weight: w_daily = 1.0, w_nse = 3650.0
- Emphasize timing: w_daily = 10.0, w_nse = 1.0
- Emphasize statistics: w_daily = 1.0, w_nse = 10000.0

Common approaches:
1. Equal contribution: Scale weights so each group contributes equally to Φ
2. Multi-objective: Treat different aspects separately, find Pareto front
3. Calibration goals: Weight based on what matters for your application

CRITICAL: DATA CONSISTENCY BETWEEN PST AND FORWARD RUN
======================================================

This is essential to understand! The PEST workflow has TWO PHASES that MUST 
use IDENTICAL observed data:

**Phase 1: PST Creation (pest_workflow.py)**

When you call `build_observations_from_data()` or `build_all_observations()`:
1. Observation NAMES are created from observed dates (e.g., "st1_20050315")
2. Observation VALUES become the TARGETS (what PEST tries to match)
3. Statistics targets are calculated (e.g., "st1_nse" with target 0.7)

**Phase 2: Forward Run (forward_run.py)**

During each PEST iteration, the forward model:
1. Runs HSPF with new parameter values
2. Loads the SAME observed data
3. Calculates SIMULATED values for the SAME observation names
4. Writes these values for PEST to read

**Why must the data be identical?**

If observed data differs between phases:
- PST has "st1_20050315" expecting value 100.5
- Forward run doesn't have 2005 data, can't produce that observation
- PEST can't compute residuals → calibration fails!

**Solution:**

Use `save_observed_data_for_forward_run()` to save the exact observed data 
used during PST creation. The forward_model.py script will load this file.

Workflow Overview
-----------------
The typical workflow for PEST++ with HSPF models involves:

1. **Setup Phase**:
   - Define parameters to estimate (e.g., LZSN, INFILT, UZSN for hydrology)
   - Create template files (TPL) that mark parameter locations in UCI file
   - Define observation locations and weights
   - Create instruction files (INS) to read model outputs

2. **PEST Control File Creation**:
   - pyemu creates a .pst file containing all PEST setup information
   - Define parameter bounds, transformations, and groups
   - Define observation groups and weights
   - Set algorithm options

3. **IMPORTANT: Save observed data for forward runs!**
   - Call `save_observed_data_for_forward_run()` with the SAME observed data
   - This ensures forward_run.py uses identical data to PST creation

4. **Run PEST++**:
   - Execute PESTPP-GLM, PESTPP-IES, or other PEST++ programs
   - Monitor convergence and adjust if needed

5. **Post-Processing**:
   - Analyze parameter sensitivity
   - Evaluate calibration results
   - Quantify prediction uncertainty

Dependencies
------------
- pyemu: Python package for PEST++ interfacing
- hspf: MPCA HSPF package with UCI class
- pandas, numpy: Data manipulation

References
----------
- PEST++ Documentation: https://github.com/usgs/pestpp
- pyemu Documentation: https://github.com/pypest/pyemu
- HSPF Manual: EPA HSPF documentation

@author: pyhcal contributors
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

try:
    import pyemu
except ImportError:
    pyemu = None

# Import metrics functions from metrics module
from pyhcal.metrics import (
    calculate_all_statistics,
    METRIC_MONTH_NAMES,
    METRIC_SEASONS,
)

# Import observation builder functions for cleaner separation of concerns
from pyhcal.observation_builder import (
    align_timeseries,
    build_timeseries_observations,
    build_statistic_observations,
    build_exceedance_observations,
    build_observations,
    ObservationBuilder,
    DEFAULT_TARGETS,
    MONTH_NAMES,
    SEASONS,
)

try:
    import pyemu
except ImportError:
    pyemu = None


class PestWorkflow:
    """
    A class to set up and manage PEST++ workflows for HSPF model calibration.
    
    Parameters
    ----------
    project_path : str or Path
        Path to the calibration project folder
    uci : object
        UCI object from the hspf package
    """
    
    # Common HSPF hydrological parameters with typical bounds
    HYDROLOGY_PARAMETERS = {
        'LZSN': {
            'block': 'PERLND',
            'table': 'PWAT-PARM2',
            'column': 'LZSN',
            'lower': 2.0,
            'upper': 15.0,
            'description': 'Lower zone nominal soil moisture storage (inches)'
        },
        'INFILT': {
            'block': 'PERLND',
            'table': 'PWAT-PARM2',
            'column': 'INFILT',
            'lower': 0.001,
            'upper': 0.5,
            'description': 'Index to infiltration capacity (in/hr)'
        },
        'AGWRC': {
            'block': 'PERLND',
            'table': 'PWAT-PARM2',
            'column': 'AGWRC',
            'lower': 0.85,
            'upper': 0.999,
            'description': 'Ground water recession parameter (per day)'
        },
                'DEEPFR': {
            'block': 'PERLND',
            'table': 'PWAT-PARM3',
            'column': 'DEEPFR',
            'lower': 0.0,
            'upper': 0.5,
            'description': 'Fraction of GW inflow to deep recharge'
        },
        'BASETP': {
            'block': 'PERLND',
            'table': 'PWAT-PARM3',
            'column': 'BASETP',
            'lower': 0.0,
            'upper': 0.2,
            'description': 'Fraction of remaining ET from baseflow'
        },
        'AGWETP': {
            'block': 'PERLND',
            'table': 'PWAT-PARM3',
            'column': 'AGWETP',
            'lower': 0.0,
            'upper': 0.2,
            'description': 'Fraction of remaining ET from active GW'
        },
        'UZSN': {
            'block': 'PERLND',
            'table': 'PWAT-PARM4',
            'column': 'UZSN',
            'lower': 0.05,
            'upper': 2.0,
            'description': 'Upper zone nominal soil moisture storage (inches)'
        },
        'INTFW': {
            'block': 'PERLND',
            'table': 'PWAT-PARM4',
            'column': 'INTFW',
            'lower': 0.0,
            'upper': 10.0,
            'description': 'Interflow inflow parameter'
        },
        'IRC': {
            'block': 'PERLND',
            'table': 'PWAT-PARM4',
            'column': 'IRC',
            'lower': 0.3,
            'upper': 0.85,
            'description': 'Interflow recession parameter (per day)'
        },
        'LZETP': {
            'block': 'PERLND',
            'table': 'PWAT-PARM4',
            'column': 'LZETP',
            'lower': 0.0,
            'upper': 0.9,
            'description': 'Lower zone ET parameter (fraction)'
        },
    }
    
    def __init__(self, project_path: Union[str, Path], uci=None):
        """
        Initialize the PEST workflow.
        
        Parameters
        ----------
        project_path : str or Path
            Path to the calibration project folder. A 'pest' subdirectory
            will be created for PEST-related files.
        uci : object, optional
            UCI object from the hspf package. If None, must be set before
            calling methods that require it.
        """
        if pyemu is None:
            raise ImportError(
                "pyemu is required for PEST++ workflows. "
                "Install it with: pip install pyemu"
            )
        
        self.project_path = Path(project_path)
        self.pest_path = self.project_path / 'pest'
        self.pest_path.mkdir(exist_ok=True)
        
        self.uci = uci
        self.parameters: Dict[str, dict] = {}
        self.observations: Dict[str, dict] = {}
        self.template_files: List[Path] = []
        self.instruction_files: List[Path] = []
        self.pst: Optional[pyemu.Pst] = None
        
    def add_parameter(
        self,
        name: str,
        block: str,
        table: str,
        table_id: int = 0,
        column: Optional[str] = None,
        opnids: Optional[List[int]] = None,
        lower_bound: float = 0.001,
        upper_bound: float = 1000.0,
        initial_value: Optional[float] = None,
        transform: str = 'log',
        group: Optional[str] = None,
        description: str = ''
    ) -> None:
        """Add a parameter definition for PEST calibration."""
        if len(name) > 12:
            raise ValueError(f"Parameter name '{name}' exceeds PEST limit of 12 characters")
        
        if column is None:
            column = name
            
        if group is None:
            group = block.lower()
        
        self.parameters[name] = {
            'block': block,
            'table': table,
            'table_id': table_id,
            'column': column,
            'opnids': opnids,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'initial_value': initial_value,
            'transform': transform,
            'group': group,
            'description': description
        }
        
    def add_hydrology_parameters(
        self,
        parameters: Optional[List[str]] = None,
        opnids: Optional[List[int]] = None,
        group: Optional[str] = None
    ) -> None:
        """Add common hydrological parameters for calibration."""
        if parameters is None:
            parameters = ['LZSN', 'INFILT', 'UZSN', 'INTFW', 'IRC', 'AGWRC']
            
        if group is None:
            group = 'hydrology'
            
        for param_name in parameters:
            if param_name not in self.HYDROLOGY_PARAMETERS:
                available = list(self.HYDROLOGY_PARAMETERS.keys())
                raise ValueError(
                    f"Unknown hydrology parameter '{param_name}'. "
                    f"Available: {available}"
                )
            
            param_info = self.HYDROLOGY_PARAMETERS[param_name]
            self.add_parameter(
                name=param_name,
                block=param_info['block'],
                table=param_info['table'],
                column=param_info['column'],
                opnids=opnids,
                lower_bound=param_info['lower'],
                upper_bound=param_info['upper'],
                group=group,
                description=param_info['description']
            )
            
    def add_observation(
        self,
        name: str,
        obs_type: str,
        station_id: Optional[str] = None,
        reach_id: Optional[int] = None,
        constituent: str = 'Q',
        time_step: str = 'D',
        weight: float = 1.0,
        group: Optional[str] = None,
        description: str = ''
    ) -> None:
        """Add observation metadata for PEST calibration."""
        if len(name) > 12:
            raise ValueError(f"Observation name '{name}' exceeds PEST limit of 12 characters")
            
        if group is None:
            group = constituent.lower()
            
        self.observations[name] = {
            'obs_type': obs_type,
            'station_id': station_id,
            'reach_id': reach_id,
            'constituent': constituent,
            'time_step': time_step,
            'weight': weight,
            'group': group,
            'description': description
        }
    
    def build_observations_from_data(
        self,
        observed: pd.Series,
        simulated: pd.Series,
        obs_group_prefix: str = 'obs',
        include_timeseries: bool = True,
        include_statistics: bool = True,
        include_exceedance: bool = False,
        statistics: Optional[List[str]] = None,
        exceedance_probs: Optional[List[float]] = None,
        timeseries_weight: float = 1.0,
        statistics_weight: float = 10.0,
        exceedance_weight: float = 5.0,
        target_values: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Build PEST observations from overlapping observed and simulated data.
        
        This method delegates to the observation_builder module for actual 
        construction, keeping the PestWorkflow class focused on orchestration.
        
        See Also
        --------
        pyhcal.observation_builder.build_observations : The underlying implementation
        """
        # Delegate to the observation_builder module
        obs_df = build_observations(
            observed=observed,
            simulated=simulated,
            prefix=obs_group_prefix,
            include_timeseries=include_timeseries,
            include_statistics=include_statistics,
            include_exceedance=include_exceedance,
            statistics=statistics,
            exceedance_probs=exceedance_probs,
            timeseries_weight=timeseries_weight,
            statistics_weight=statistics_weight,
            exceedance_weight=exceedance_weight,
            target_values=target_values,
            verbose=True
        )
        
        # Store internally
        self._observation_data = obs_df
        
        return obs_df
    
    def build_observations_for_location(
        self,
        obs_name: str,
        observed: pd.Series,
        simulated: pd.Series,
        include_timeseries: bool = True,
        include_statistics: bool = True,
        include_exceedance: bool = False,
        statistics: Optional[List[str]] = None,
        exceedance_probs: Optional[List[float]] = None,
        target_values: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Build PEST observations for a specific location defined via add_observation()."""
        # Get metadata for this observation
        if obs_name not in self.observations:
            available = list(self.observations.keys())
            raise ValueError(
                f"Observation '{obs_name}' not found. "
                f"Define it first with add_observation(). "
                f"Available observations: {available}"
            )
        
        metadata = self.observations[obs_name]
        
        # Use metadata to configure the build
        base_weight = metadata['weight']
        base_group = metadata['group']  # This will be used for PEST obgnme
        
        print(f"\nBuilding observations for '{obs_name}':")
        print(f"  Station ID: {metadata['station_id']}")
        print(f"  Reach ID: {metadata['reach_id']}")
        print(f"  Constituent: {metadata['constituent']}")
        print(f"  Base weight: {base_weight}")
        print(f"  PEST group (obgnme): {base_group}")
        
        # Build observations using the obs_name as prefix for obs names
        obs_df = self.build_observations_from_data(
            observed=observed,
            simulated=simulated,
            obs_group_prefix=obs_name,
            include_timeseries=include_timeseries,
            include_statistics=include_statistics,
            include_exceedance=include_exceedance,
            statistics=statistics,
            exceedance_probs=exceedance_probs,
            timeseries_weight=base_weight,  # Use metadata weight!
            statistics_weight=base_weight * 10.0,  # Scale statistics weight
            exceedance_weight=base_weight * 5.0,  # Scale exceedance weight
            target_values=target_values
        )
        
        # POST-PROCESS: Replace auto-generated obgnme with metadata group
        # The auto-generated groups are like: obs_name_ts, obs_name_stat, etc.
        # We replace them with: group_ts, group_stat, etc.
        # This way, the metadata 'group' is actually used in the PEST file!
        obs_df['obgnme'] = obs_df['obgnme'].str.replace(
            f'^{obs_name}_', f'{base_group}_', regex=True
        )
        
        # Store in a dict keyed by observation name
        if not hasattr(self, '_location_observations'):
            self._location_observations = {}
        self._location_observations[obs_name] = obs_df
        
        return obs_df
    
    def build_all_observations(
        self,
        data_dict: Dict[str, Tuple[pd.Series, pd.Series]],
        include_timeseries: bool = True,
        include_statistics: bool = True,
        include_exceedance: bool = False,
        statistics: Optional[List[str]] = None,
        exceedance_probs: Optional[List[float]] = None,
        target_values: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Build observations for ALL locations defined via add_observation()."""
        all_observations = []
        
        # Validate that all data_dict keys exist in observations
        for obs_name in data_dict.keys():
            if obs_name not in self.observations:
                available = list(self.observations.keys())
                raise ValueError(
                    f"Observation '{obs_name}' in data_dict not found in defined observations. "
                    f"Define it first with add_observation(). "
                    f"Available: {available}"
                )
        
        # Build observations for each location
        for obs_name, (observed, simulated) in data_dict.items():
            obs_df = self.build_observations_for_location(
                obs_name=obs_name,
                observed=observed,
                simulated=simulated,
                include_timeseries=include_timeseries,
                include_statistics=include_statistics,
                include_exceedance=include_exceedance,
                statistics=statistics,
                exceedance_probs=exceedance_probs,
                target_values=target_values
            )
            all_observations.append(obs_df)
        
        # Combine all observations
        combined_df = pd.concat(all_observations, ignore_index=True)
        
        # Store combined result
        self._all_observation_data = combined_df
        
        print(f"\n{'='*60}")
        print(f"COMBINED OBSERVATIONS FROM ALL LOCATIONS")
        print(f"{'='*60}")
        print(f"Total observations: {len(combined_df)}")
        print(f"\nBy location:")
        location_counts = combined_df.groupby(
            combined_df['obgnme'].str.rsplit('_', n=1).str[0]
        ).size()
        for loc, count in location_counts.items():
            weight = self.observations.get(loc, {}).get('weight', 'N/A')
            print(f"  {loc}: {count} observations (weight: {weight})")
        
        return combined_df
    
    def get_all_observation_data(self) -> pd.DataFrame:
        """
        Get the combined observation data from all locations.
        
        Returns
        -------
        pd.DataFrame
            Combined observation data from build_all_observations() or 
            concatenation of build_observations_for_location() results.
        """
        if hasattr(self, '_all_observation_data'):
            return self._all_observation_data
        elif hasattr(self, '_location_observations'):
            return pd.concat(self._location_observations.values(), ignore_index=True)
        elif hasattr(self, '_observation_data'):
            return self._observation_data
        else:
            raise ValueError(
                "No observation data found. Call build_observations_from_data(), "
                "build_observations_for_location(), or build_all_observations() first."
            )
    
    def save_observed_data_for_forward_run(
        self,
        observed_data_dict: Dict[str, pd.Series],
        output_file: Optional[str] = None
    ) -> Path:
        """
        Save observed data to CSV for use by the forward_run script.
        
        CRITICAL: This ensures consistency between PST creation and forward runs!
        
        The forward_run script MUST use the exact same observed data that was
        used to create the PEST control file. Otherwise:
        - Observation names won't match (different dates → different obsnme)
        - Statistics will be computed on different data subsets
        - Calibration will fail or produce incorrect results
        
        Parameters
        ----------
        observed_data_dict : dict
            Dictionary mapping observation names to observed Series.
            Keys should match names used in add_observation() and data_dict
            used in build_all_observations().
        output_file : str, optional
            Output filename. If None, uses 'observed_data.csv' in pest_path.
            
        Returns
        -------
        Path
            Path to the saved CSV file
            
        Notes
        -----
        The saved CSV has this format:
        - 'date' column: DatetimeIndex as strings
        - One column per observation location (obs_name)
        
        Example::
        
            date,station1,station2,station3
            2005-01-01,100.5,45.2,78.9
            2005-01-02,105.3,48.1,82.4
            ...
            
        The forward_run.py script loads this file and uses it to:
        1. Get observed values for timeseries comparisons
        2. Calculate statistics (NSE, etc.) from overlapping obs/sim
        3. Produce simulated values for the SAME observation names in PST
        
        Examples
        --------
        Save observed data after building observations::
        
            # First, define observations and build PST data
            pest.add_observation('st1', 'timeseries', reach_id=45)
            pest.add_observation('st2', 'timeseries', reach_id=67)
            
            data_dict = {
                'st1': (obs_series_st1, sim_series_st1),
                'st2': (obs_series_st2, sim_series_st2),
            }
            pest.build_all_observations(data_dict)
            
            # IMPORTANT: Save the observed data for forward_run consistency
            obs_data = {
                'st1': obs_series_st1,
                'st2': obs_series_st2,
            }
            pest.save_observed_data_for_forward_run(obs_data)
            
            # Now create forward model script (it will use this saved data)
            pest.create_forward_model_script()
        """
        if output_file is None:
            output_file = 'observed_data.csv'
        
        output_path = self.pest_path / output_file
        
        # Combine all observed series into a DataFrame
        # Each series becomes a column, index becomes date column
        combined_df = pd.DataFrame(observed_data_dict)
        combined_df.index.name = 'date'
        
        # Save to CSV
        combined_df.to_csv(output_path)
        
        print(f"\nObserved data saved for forward_run consistency:")
        print(f"  File: {output_path}")
        print(f"  Locations: {list(observed_data_dict.keys())}")
        print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"  Total records per location:")
        for col in combined_df.columns:
            non_null = combined_df[col].notna().sum()
            print(f"    {col}: {non_null} values")
        
        # Also save metadata about what was included
        self._saved_observed_file = output_path
        
        return output_path
    
    def calculate_model_statistics(
        self,
        observed: pd.Series,
        simulated: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate statistics from overlapping observed and simulated data.
        
        This is what the model post-processor should compute and write
        to a file for PEST to read via instruction file.
        
        This method is a convenience wrapper around the functions in 
        pyhcal.metrics module.
        
        Parameters
        ----------
        observed : pd.Series
            Observed data with datetime index
        simulated : pd.Series
            Simulated data with datetime index
            
        Returns
        -------
        dict
            Dictionary of computed statistics
            
        Notes
        -----
        These statistics should be written to a file that PEST reads
        using an instruction file (.ins). The instruction file tells
        PEST where to find each simulated value in the output file.
        
        Example output file format::
        
            nse  0.723
            kge  0.681
            bias -5.23
            pbias -3.2
            jan_avg 125.4
            feb_avg 142.1
            ...
            q10 485.2
            q50 98.3
            q90 18.7
            
        See Also
        --------
        pyhcal.metrics.calculate_all_statistics : Full implementation
        pyhcal.metrics.nse_metric : NSE calculation
        pyhcal.metrics.kge_metric : KGE calculation
        """
        # Use the centralized implementation from metrics module
        return calculate_all_statistics(
            observed, 
            simulated,
            include_monthly=True,
            include_seasonal=True,
            include_exceedance=True,
            exceedance_probs=[10, 25, 50, 75, 90]
        )
        return results
        
    def create_template_file(self, output_path: Optional[Path] = None) -> Path:
        """
        Create a PEST template file from the UCI file.
        
        This method uses the UCI class's add_parameter_template method to
        mark parameter locations in the UCI file with PEST template markers.
        
        Parameters
        ----------
        output_path : Path, optional
            Path for the template file. If None, uses pest_path with .tpl extension.
            
        Returns
        -------
        Path
            Path to the created template file
            
        Notes
        -----
        Template File Format:
        - First line: 'ptf ~' (parameter template file marker)
        - Parameter locations marked as: ~parameter_name~
        - Original file content preserved elsewhere
        
        The UCI class method add_parameter_template(BLOCK, TABLE, TABLEID, PARAMETER)
        handles the actual template creation.
        
        Examples
        --------
        Create template file::
        
            tpl_file = pest.create_template_file()
            print(f"Template file created: {tpl_file}")
        """
        if self.uci is None:
            raise ValueError("UCI object must be set before creating template files")
            
        if not self.parameters:
            raise ValueError("No parameters defined. Use add_parameter() first.")
            
        if output_path is None:
            output_path = self.pest_path / f"{self.project_path.name}.tpl"
            
        # Use the UCI class's add_parameter_template method
        # This creates template markers for each parameter
        for param_name, param_info in self.parameters.items():
            self.uci.add_parameter_template(
                param_info['block'],
                param_info['table'],
                param_info['table_id'],
                param_name,
                single_template = True
            )
            
        # Write the template file
        self.uci.write_tpl(new_tpl_path = output_path)
        self.template_files.append(output_path)
        
        return output_path
        
    def build_pst(
        self,
        name: Optional[str] = None,
        algorithm: str = 'glm',
        noptmax: int = 30,
        phiredswh: float = 0.1,
        svd_pack: str = 'propack'
    ) -> 'pyemu.Pst':
        """
        Build a PEST control file using pyemu.
        
        This method creates a complete PEST control file (.pst) that can be
        used with PEST++ programs for parameter estimation.
        
        Parameters
        ----------
        name : str, optional
            Name for the PEST case. If None, uses project name.
        algorithm : str, default 'glm'
            PEST++ algorithm to configure for:
            - 'glm': Gradient-based (PESTPP-GLM)
            - 'ies': Ensemble smoother (PESTPP-IES)
            - 'sen': Sensitivity analysis (PESTPP-SEN)
        noptmax : int, default 30
            Maximum number of optimization iterations
        phiredswh : float, default 0.1
            Relative objective function reduction for switching from
            quasi-Newton to full Newton search
        svd_pack : str, default 'propack'
            SVD package: 'propack' (fast) or 'eigen' (accurate)
            
        Returns
        -------
        pyemu.Pst
            The PEST control file object
            
        Notes
        -----
        PEST Control File Sections:
        - * control data: Algorithm settings
        - * parameter groups: Groups and their regularization
        - * parameter data: Individual parameter definitions
        - * observation groups: Groups for observations
        - * observation data: Individual observation definitions
        - * model command line: Command to run the model
        - * model input/output: Template and instruction files
        
        Examples
        --------
        Build and customize PEST control file::
        
            pst = pest.build_pst(algorithm='glm', noptmax=50)
            
            # Access and modify parameter data
            par = pst.parameter_data
            par.loc['LZSN', 'parlbnd'] = 3.0  # Adjust bounds
            
            # Access and modify observation data
            obs = pst.observation_data
        """
        if name is None:
            name = self.project_path.name
            
        # Create pyemu Pst object
        # This is a simplified version - full implementation would use
        # pyemu.helpers or pyemu.PstFrom for more complex setups
        
        # Create parameter dataframe
        par_data = []
        for param_name, param_info in self.parameters.items():
            initial = param_info['initial_value']
            if initial is None:
                # Would get from UCI table - using midpoint as placeholder
                initial = (param_info['lower_bound'] + param_info['upper_bound']) / 2
                
            par_data.append({
                'parnme': param_name,
                'partrans': param_info['transform'],
                'parchglim': 'relative',
                'parval1': initial,
                'parlbnd': param_info['lower_bound'],
                'parubnd': param_info['upper_bound'],
                'pargp': param_info['group'],
                'scale': 1.0,
                'offset': 0.0,
                'dercom': 1
            })
            
        par_df = pd.DataFrame(par_data)
        par_df.set_index('parnme', inplace=True)
        
        # Create observation dataframe (placeholder structure)
        obs_data = []
        for obs_name, obs_info in self.observations.items():
            obs_data.append({
                'obsnme': obs_name,
                'obsval': 0.0,  # Would be filled from actual data
                'weight': obs_info['weight'],
                'obgnme': obs_info['group']
            })
            
        if obs_data:
            obs_df = pd.DataFrame(obs_data)
            obs_df.set_index('obsnme', inplace=True)
        else:
            # Create minimal observation if none defined
            obs_df = pd.DataFrame({
                'obsnme': ['placeholder'],
                'obsval': [0.0],
                'weight': [1.0],
                'obgnme': ['obs']
            })
            obs_df.set_index('obsnme', inplace=True)
        
        # Create PST object
        self.pst = pyemu.Pst.from_par_obs_names(
            par_names=list(self.parameters.keys()),
            obs_names=list(obs_df.index)
        )
        
        # Update parameter data - need to update existing columns, not replace
        pst_par = self.pst.parameter_data
        for param_name, param_info in self.parameters.items():
            initial = param_info['initial_value']
            if initial is None:
                initial = (param_info['lower_bound'] + param_info['upper_bound']) / 2
            
            pst_par.loc[param_name, 'partrans'] = param_info['transform']
            pst_par.loc[param_name, 'parchglim'] = 'relative'
            pst_par.loc[param_name, 'parval1'] = initial
            pst_par.loc[param_name, 'parlbnd'] = param_info['lower_bound']
            pst_par.loc[param_name, 'parubnd'] = param_info['upper_bound']
            pst_par.loc[param_name, 'pargp'] = param_info['group']
            pst_par.loc[param_name, 'scale'] = 1.0
            pst_par.loc[param_name, 'offset'] = 0.0
        
        # Update observation data
        pst_obs = self.pst.observation_data
        for obs_name, obs_info in self.observations.items():
            pst_obs.loc[obs_name, 'obsval'] = 0.0
            pst_obs.loc[obs_name, 'weight'] = obs_info['weight']
            pst_obs.loc[obs_name, 'obgnme'] = obs_info['group']
        
        # Set control data
        self.pst.control_data.noptmax = noptmax
        self.pst.control_data.phiredswh = phiredswh
        self.pst.svd_data.svdpack = svd_pack
        
        return self.pst
    
    def build_pst_from_observation_data(
        self,
        name: Optional[str] = None,
        model_command: str = 'python run_model.py',
        tpl_file: Optional[str] = None,
        in_file: Optional[str] = None,
        ins_file: Optional[str] = None,
        out_file: Optional[str] = None,
        algorithm: str = 'glm',
        noptmax: int = 30
    ) -> 'pyemu.Pst':
        """
        Build a PEST control file using the ACTUAL observation data.
        
        THIS IS THE CORRECT METHOD TO USE after calling build_all_observations()
        or build_observations_from_data(). It properly transfers all observation
        data (names, values, weights, groups) to the PST object.
        
        THE PEST WORKFLOW
        =================
        
        For PEST to work, it needs to:
        1. **Read parameters** from a template file (.tpl) 
        2. **Run the model** using a command line
        3. **Extract simulated values** from model output using an instruction file (.ins)
        4. **Calculate residuals** between observed and simulated values
        5. **Update parameters** to minimize residuals
        
        Files needed:
        - .tpl file: Template with parameter placeholders (e.g., ~LZSN~)
        - .pst file: PEST control file with all settings
        - .ins file: Instruction file telling PEST how to read model output
        - Model output file: Where simulated values are written
        - Forward model script: Runs model and writes output in expected format
        
        Parameters
        ----------
        name : str, optional
            Name for the PEST case. If None, uses project name.
        model_command : str, default 'python run_model.py'
            Command to run the forward model. This should run HSPF and
            then extract/calculate the simulated values.
        tpl_file : str, optional
            Template file path (relative to pest_path). If None, uses {name}.tpl
        in_file : str, optional  
            Input file that template creates (e.g., model.uci). If None, uses {name}.uci
        ins_file : str, optional
            Instruction file path. If None, uses {name}.ins
        out_file : str, optional
            Model output file path. If None, uses {name}_output.txt
        algorithm : str, default 'glm'
            PEST++ algorithm ('glm', 'ies', 'sen')
        noptmax : int, default 30
            Maximum optimization iterations
            
        Returns
        -------
        pyemu.Pst
            Complete PEST control file with all observation data
            
        Examples
        --------
        Complete workflow::
        
            # 1. Add parameters
            pest.add_hydrology_parameters(['LZSN', 'INFILT', 'AGWRC'])
            
            # 2. Add observation metadata
            pest.add_observation(name='flow', obs_type='timeseries',
                               reach_id=650, group='q', weight=1.0)
            
            # 3. Build observations from data
            all_obs = pest.build_all_observations(
                data_dict={'flow': (obs_data, sim_data)},
                include_statistics=True
            )
            
            # 4. Create instruction file
            pest.create_instruction_file()
            
            # 5. Create forward model script
            pest.create_forward_model_script()
            
            # 6. Build PST with ALL observation data
            pst = pest.build_pst_from_observation_data(
                model_command='python forward_model.py'
            )
            
            # 7. Verify observations are in PST
            print(f"Observations in PST: {len(pst.observation_data)}")
        """
        if name is None:
            name = self.project_path.name
        
        # Get observation data
        obs_df = self.get_all_observation_data()
        
        if obs_df is None or len(obs_df) == 0:
            raise ValueError(
                "No observation data found. Call build_observations_from_data() "
                "or build_all_observations() first."
            )
        
        # Set up file paths
        if tpl_file is None:
            tpl_file = f"{name}.tpl"
        if in_file is None:
            in_file = f"{name}.uci"
        if ins_file is None:
            ins_file = f"{name}.ins"
        if out_file is None:
            out_file = f"{name}_output.txt"
        
        print(f"Building PST with {len(obs_df)} observations and {len(self.parameters)} parameters")
        
        # Create PST from parameter and observation names
        par_names = list(self.parameters.keys())
        obs_names = list(obs_df['obsnme'])
        
        self.pst = pyemu.Pst.from_par_obs_names(
            par_names=par_names,
            obs_names=obs_names
        )
        
        # Update parameter data
        pst_par = self.pst.parameter_data
        for param_name, param_info in self.parameters.items():
            initial = param_info['initial_value']
            if initial is None:
                initial = (param_info['lower_bound'] + param_info['upper_bound']) / 2
            
            pst_par.loc[param_name, 'partrans'] = param_info['transform']
            pst_par.loc[param_name, 'parchglim'] = 'relative'
            pst_par.loc[param_name, 'parval1'] = initial
            pst_par.loc[param_name, 'parlbnd'] = param_info['lower_bound']
            pst_par.loc[param_name, 'parubnd'] = param_info['upper_bound']
            pst_par.loc[param_name, 'pargp'] = param_info['group']
            pst_par.loc[param_name, 'scale'] = 1.0
            pst_par.loc[param_name, 'offset'] = 0.0
        
        # Update observation data FROM THE ACTUAL DATA!
        pst_obs = self.pst.observation_data
        for _, row in obs_df.iterrows():
            obs_name = row['obsnme']
            pst_obs.loc[obs_name, 'obsval'] = row['obsval']
            pst_obs.loc[obs_name, 'weight'] = row['weight']
            pst_obs.loc[obs_name, 'obgnme'] = row['obgnme']
        
        # Set model I/O using the model_input_data and model_output_data properties
        self.pst.model_command = [model_command]
        
        # Add template/input files via the model_input_data DataFrame
        # pyemu uses DataFrames for these, not simple lists
        input_data = pd.DataFrame({
            'pest_file': [tpl_file],
            'model_file': [in_file]
        })
        self.pst.model_input_data = input_data
        
        # Add instruction/output files via the model_output_data DataFrame
        output_data = pd.DataFrame({
            'pest_file': [ins_file],
            'model_file': [out_file]
        })
        self.pst.model_output_data = output_data
        
        # Set control data
        self.pst.control_data.noptmax = noptmax
        
        # Store file info for later use
        self._pest_files = {
            'tpl_file': tpl_file,
            'in_file': in_file,
            'ins_file': ins_file,
            'out_file': out_file,
            'model_command': model_command
        }
        
        print(f"PST created successfully:")
        print(f"  Parameters: {len(pst_par)}")
        print(f"  Observations: {len(pst_obs)}")
        print(f"  Observation groups: {pst_obs['obgnme'].nunique()}")
        
        return self.pst
    
    def create_instruction_file(
        self,
        output_path: Optional[Path] = None,
        obs_df: Optional[pd.DataFrame] = None
    ) -> Path:
        """
        Create a PEST instruction file (.ins) for reading model output.
        
        WHAT IS AN INSTRUCTION FILE?
        ============================
        
        An instruction file tells PEST how to extract simulated values from
        model output files. PEST reads this file after each model run to
        get the simulated values corresponding to each observation.
        
        The instruction file format uses special markers:
        - `pif ~` : First line, defines the marker character (~)
        - `l1` : Skip to line 1
        - `~obs_name~` : Read a value and call it obs_name
        
        Parameters
        ----------
        output_path : Path, optional
            Path for instruction file. If None, uses pest_path/{name}.ins
        obs_df : pd.DataFrame, optional
            Observation data. If None, uses get_all_observation_data()
            
        Returns
        -------
        Path
            Path to created instruction file
            
        Notes
        -----
        The instruction file expects model output in this format::
        
            obs_name1  value1
            obs_name2  value2
            ...
        
        Your forward model script must write output in this exact format.
        See `create_forward_model_script()` for an example.
        """
        if obs_df is None:
            obs_df = self.get_all_observation_data()
        
        if obs_df is None or len(obs_df) == 0:
            raise ValueError(
                "No observation data. Call build_observations_from_data() first."
            )
        
        name = self.project_path.name
        if output_path is None:
            output_path = self.pest_path / f"{name}.ins"
        
        # Create instruction file content
        lines = ['pif ~']  # Marker character is ~
        
        for i, row in obs_df.iterrows():
            obs_name = row['obsnme']
            # Instruction: skip to next line, read whitespace-separated value
            # Format: l1 !obs_name! reads a free-format value
            lines.append(f'l1 !{obs_name}!')
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self.instruction_files.append(output_path)
        print(f"Instruction file created: {output_path}")
        print(f"  Contains {len(obs_df)} observation instructions")
        
        return output_path
    
    def create_forward_model_script(
        self,
        output_path: Optional[Path] = None,
        model_exe: str = 'hspf',
        uci_file: Optional[str] = None,
        observed_data_file: Optional[str] = None
    ) -> Path:
        """Create a Python script that runs the forward model for PEST."""
        name = self.project_path.name
        if output_path is None:
            output_path = self.pest_path / 'forward_model.py'
        if uci_file is None:
            uci_file = f'{name}.uci'
        
        output_file = f'{name}_output.txt'
        obs_data_file = observed_data_file or 'observed_data.csv'
        
        # Build reach_map and observation_info dictionaries
        reach_map_str = "{\n"
        obs_info_str = "{\n"
        for obs_name, obs_info in self.observations.items():
            reach_id = obs_info.get('reach_id', 'None')
            reach_map_str += f"    '{obs_name}': {reach_id},\n"
            
            # Build observation info dict
            obs_info_str += f"    '{obs_name}': {{\n"
            obs_info_str += f"        'constituent': '{obs_info.get('constituent', 'ROVOL')}',\n"
            obs_info_str += f"        'obs_column': '{obs_info.get('obs_column', obs_name)}',\n"
            obs_info_str += f"    }},\n"
        reach_map_str += "}"
        obs_info_str += "}"
        
        # Generate the forward model script using the forward_run module
        script = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEST Forward Model Script for {name}

This script uses the pyhcal.forward_run module for model execution.
Customize extract_simulated_data() in forward_run.py for your model output format.
"""

from pyhcal.forward_run import (
    ForwardModelConfig,
    run_forward_model,
)

# ===========================================================================
# CONFIGURATION
# ===========================================================================

config = ForwardModelConfig(
    model_exe='{model_exe}',
    uci_file='{uci_file}',
    output_file='{output_file}',
    observed_data_file='{obs_data_file}',
    reach_map={reach_map_str},
    observation_info={obs_info_str},
)

# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    run_forward_model(config)
'''
        
        # Write script
        with open(output_path, 'w') as f:
            f.write(script)
        
        print(f"Forward model script created: {output_path}")
        
        return output_path
    
    def write_pest_files(self, pest_case_name: Optional[str] = None) -> Path:
        """
        Write all PEST files to the pest directory.
        
        Parameters
        ----------
        pest_case_name : str, optional
            Base name for PEST files. If None, uses project name.
            
        Returns
        -------
        Path
            Path to the main PEST control file (.pst)
            
        Notes
        -----
        Files Created:
        - {name}.pst: PEST control file
        - {name}.tpl: Template file(s) for model input
        - {name}.ins: Instruction file(s) for model output
        
        Examples
        --------
        Write PEST files::
        
            pst_path = pest.write_pest_files()
            print(f"PEST control file: {pst_path}")
            print("Run PEST++ with: pestpp-glm {pst_path}")
        """
        if self.pst is None:
            raise ValueError("Must call build_pst() before writing files")
            
        if pest_case_name is None:
            pest_case_name = self.project_path.name
            
        pst_path = self.pest_path / f"{pest_case_name}.pst"
        self.pst.write(str(pst_path))
        
        return pst_path
        
    def run_sensitivity_analysis(
        self,
        method: str = 'morris',
        num_trajectories: int = 10
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis using PEST++.
        
        Parameters
        ----------
        method : str, default 'morris'
            Sensitivity analysis method:
            - 'morris': Method of Morris (screening method)
            - 'sobol': Sobol variance-based method
        num_trajectories : int, default 10
            Number of trajectories for Morris method
            
        Returns
        -------
        pd.DataFrame
            Sensitivity analysis results
            
        Notes
        -----
        Sensitivity Analysis Methods:
        
        Method of Morris:
        - Elementary effects method (screening)
        - Computes mean and standard deviation of parameter effects
        - Parameters with high mean are influential
        - Parameters with high std have non-linear or interaction effects
        
        Sobol Method:
        - Variance-based global sensitivity
        - Decomposes output variance by parameter contributions
        - First-order (S1): Individual parameter effect
        - Total order (ST): Including all interactions
        
        Examples
        --------
        Run Morris sensitivity analysis::
        
            sensitivity = pest.run_sensitivity_analysis(method='morris')
            print(sensitivity.sort_values('mean_effect', ascending=False))
        """
        if self.pst is None:
            raise ValueError("Must call build_pst() before running sensitivity analysis")
            
        # Configure for sensitivity analysis
        # This would typically involve running PESTPP-SEN
        # Here we show the setup - actual execution requires PEST++ binaries
        
        self.pst.pestpp_options['gsa_method'] = method
        if method == 'morris':
            self.pst.pestpp_options['gsa_morris_r'] = num_trajectories
            
        # Return placeholder - actual results come from PESTPP-SEN run
        return pd.DataFrame({
            'parameter': list(self.parameters.keys()),
            'mean_effect': np.nan,
            'std_effect': np.nan,
            'note': 'Run PESTPP-SEN to populate results'
        })
        
    def get_parameter_summary(self) -> pd.DataFrame:
        """
        Get a summary of all defined parameters.
        
        Returns
        -------
        pd.DataFrame
            Summary of parameter definitions including bounds and groups
            
        Examples
        --------
        View parameter summary::
        
            summary = pest.get_parameter_summary()
            print(summary)
        """
        data = []
        for name, info in self.parameters.items():
            data.append({
                'name': name,
                'block': info['block'],
                'table': info['table'],
                'column': info['column'],
                'lower': info['lower_bound'],
                'upper': info['upper_bound'],
                'transform': info['transform'],
                'group': info['group'],
                'description': info['description']
            })
        return pd.DataFrame(data)
        
    def get_observation_summary(self) -> pd.DataFrame:
        """
        Get a summary of all defined observations.
        
        Returns
        -------
        pd.DataFrame
            Summary of observation definitions
            
        Examples
        --------
        View observation summary::
        
            summary = pest.get_observation_summary()
            print(summary)
        """
        data = []
        for name, info in self.observations.items():
            data.append({
                'name': name,
                'type': info['obs_type'],
                'station_id': info['station_id'],
                'reach_id': info['reach_id'],
                'constituent': info['constituent'],
                'weight': info['weight'],
                'group': info['group']
            })
        return pd.DataFrame(data)


def create_pest_workflow(
    project_path: Union[str, Path],
    uci=None
) -> PestWorkflow:
    """
    Factory function to create a PEST workflow instance.
    
    This is the recommended way to create a PestWorkflow object.
    
    Parameters
    ----------
    project_path : str or Path
        Path to the calibration project folder
    uci : object, optional
        UCI object from the hspf package
        
    Returns
    -------
    PestWorkflow
        Initialized workflow object
        
    Examples
    --------
    Create a workflow for hydrology calibration::
    
        from pyhcal.pest_workflow import create_pest_workflow
        
        pest = create_pest_workflow('/path/to/project', uci=cal.uci)
        pest.add_hydrology_parameters()
        pest.add_observation(
            name='flow',
            obs_type='timeseries',
            reach_id=650,
            constituent='Q'
        )
        pest.build_pst()
        pest.write_pest_files()
    """
    return PestWorkflow(project_path, uci)


# Quick reference for common PEST++ commands
PEST_COMMANDS = """
PEST++ Quick Reference
======================

Running PEST++:
    pestpp-glm model.pst          # Gradient-based estimation
    pestpp-ies model.pst          # Ensemble smoother
    pestpp-sen model.pst          # Sensitivity analysis

Key PESTPP options (add to .pst file):
    ++overdue_giveup_fac(2.0)     # Skip slow model runs
    ++max_run_fail(3)             # Max consecutive failures
    ++ies_num_reals(50)           # Ensemble size for IES
    ++gsa_method(morris)          # Sensitivity method

Output files:
    model.rec                     # Main record file
    model.par                     # Estimated parameters
    model.sen                     # Sensitivity results
    model.iobj                    # Objective function history
"""
