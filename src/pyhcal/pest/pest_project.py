"""
PestProject - Simple interface for setting up PEST++ projects for HSPF calibration.

This module provides a clean, readable API for creating PEST++ projects that
integrate with the pyhcal Calibrator class. The workflow is designed to work
in-memory with pandas DataFrames that can be easily used to create pyemu.Pst objects.

Usage:
    cal.pest.create_project()
    cal.pest.add_observation(name, station_id, reach_ids, constituent, time_step, weight)
    cal.pest.add_parameter(block, table, table_id, parameter, ...)
    
    obs_df = cal.pest.build_observations()
    par_df = cal.pest.build_parameters()
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from pyhcal import Calibrator

# Default PEST parameter properties for common HSPF hydrology parameters
# Key format: (block, table, table_id, parameter)
HSPF_PARAMETER_DEFAULTS: Dict[tuple, Dict[str, Any]] = {
    # PERLND parameters - PWAT-PARM2 (table_id=0)
    ('PERLND', 'PWAT-PARM2', 0, 'LZSN'):   {'parval1': 6.0,  'parlbnd': 2.0,   'parubnd': 15.0,  'pargp': 'storage'},
    ('PERLND', 'PWAT-PARM2', 0, 'INFILT'): {'parval1': 0.10, 'parlbnd': 0.001, 'parubnd': 0.50,  'pargp': 'infiltration'},
    ('PERLND', 'PWAT-PARM2', 0, 'LSUR'):   {'parval1': 400,  'parlbnd': 100,   'parubnd': 700,   'pargp': 'surface'},
    ('PERLND', 'PWAT-PARM2', 0, 'SLSUR'):  {'parval1': 0.10, 'parlbnd': 0.001, 'parubnd': 0.30,  'pargp': 'surface'},
    ('PERLND', 'PWAT-PARM2', 0, 'KVARY'):  {'parval1': 0.0,  'parlbnd': 0.0,   'parubnd': 5.0,   'pargp': 'groundwater'},
    ('PERLND', 'PWAT-PARM2', 0, 'AGWRC'):  {'parval1': 0.98, 'parlbnd': 0.85,  'parubnd': 0.999, 'pargp': 'groundwater'},
    
    # PERLND parameters - PWAT-PARM3 (table_id=0)
    ('PERLND', 'PWAT-PARM3', 0, 'PETMAX'): {'parval1': 40.0, 'parlbnd': 32.0,  'parubnd': 48.0,  'pargp': 'et'},
    ('PERLND', 'PWAT-PARM3', 0, 'PETMIN'): {'parval1': 35.0, 'parlbnd': 30.0,  'parubnd': 40.0,  'pargp': 'et'},
    ('PERLND', 'PWAT-PARM3', 0, 'INFEXP'): {'parval1': 2.0,  'parlbnd': 1.0,   'parubnd': 10.0,  'pargp': 'infiltration'},
    ('PERLND', 'PWAT-PARM3', 0, 'INFILD'): {'parval1': 2.0,  'parlbnd': 1.0,   'parubnd': 2.0,   'pargp': 'infiltration'},
    ('PERLND', 'PWAT-PARM3', 0, 'DEEPFR'): {'parval1': 0.0,  'parlbnd': 0.0,   'parubnd': 0.50,  'pargp': 'groundwater'},
    ('PERLND', 'PWAT-PARM3', 0, 'BASETP'): {'parval1': 0.0,  'parlbnd': 0.0,   'parubnd': 0.20,  'pargp': 'et'},
    ('PERLND', 'PWAT-PARM3', 0, 'AGWETP'): {'parval1': 0.0,  'parlbnd': 0.0,   'parubnd': 0.20,  'pargp': 'et'},
    
    # PERLND parameters - PWAT-PARM4 (table_id=0)
    ('PERLND', 'PWAT-PARM4', 0, 'CEPSC'):  {'parval1': 0.1,  'parlbnd': 0.01,  'parubnd': 0.40,  'pargp': 'interception'},
    ('PERLND', 'PWAT-PARM4', 0, 'UZSN'):   {'parval1': 1.0,  'parlbnd': 0.05,  'parubnd': 5.0,   'pargp': 'storage'},
    ('PERLND', 'PWAT-PARM4', 0, 'NSUR'):   {'parval1': 0.25, 'parlbnd': 0.05,  'parubnd': 0.50,  'pargp': 'surface'},
    ('PERLND', 'PWAT-PARM4', 0, 'INTFW'):  {'parval1': 1.5,  'parlbnd': 0.0,   'parubnd': 10.0,  'pargp': 'interflow'},
    ('PERLND', 'PWAT-PARM4', 0, 'IRC'):    {'parval1': 0.70, 'parlbnd': 0.30,  'parubnd': 0.85,  'pargp': 'interflow'},
    ('PERLND', 'PWAT-PARM4', 0, 'LZETP'):  {'parval1': 0.30, 'parlbnd': 0.0,   'parubnd': 0.90,  'pargp': 'et'},
    
    # IMPLND parameters - IWAT-PARM2 (table_id=0)
    ('IMPLND', 'IWAT-PARM2', 0, 'LSUR'):   {'parval1': 400,  'parlbnd': 100,   'parubnd': 700,   'pargp': 'surface'},
    ('IMPLND', 'IWAT-PARM2', 0, 'SLSUR'):  {'parval1': 0.10, 'parlbnd': 0.001, 'parubnd': 0.30,  'pargp': 'surface'},
    ('IMPLND', 'IWAT-PARM2', 0, 'NSUR'):   {'parval1': 0.10, 'parlbnd': 0.01,  'parubnd': 0.25,  'pargp': 'surface'},
    ('IMPLND', 'IWAT-PARM2', 0, 'RETSC'):  {'parval1': 0.10, 'parlbnd': 0.01,  'parubnd': 0.30,  'pargp': 'interception'},
    
    # RCHRES parameters - HYDR-PARM2 (table_id=0)
    ('RCHRES', 'HYDR-PARM2', 0, 'FTBUCI'): {'parval1': 1.0,  'parlbnd': 0.5,   'parubnd': 2.0,   'pargp': 'routing'},
    ('RCHRES', 'HYDR-PARM2', 0, 'KS'):     {'parval1': 0.5,  'parlbnd': 0.0,   'parubnd': 1.0,   'pargp': 'routing'},
}


class PestProject:
    """
    Simple interface for setting up PEST++ projects for HSPF calibration.
    
    This class is designed to be a component of the Calibrator class, accessed
    as `cal.pest`. It provides a clean API for:
    - Creating a PEST project directory
    - Adding observations (station data to calibrate against)
    - Adding parameters (HSPF parameters to adjust)
    - Building DataFrames for use with pyemu.Pst
    
    All data is stored in memory until explicitly written to files.
    
    Attributes:
        calibrator: Reference to parent Calibrator object
        project_path: Path to pest project directory
        observations: List of observation configurations
        parameters: List of parameter configurations
    """
    
    def __init__(self, calibrator: 'Calibrator'):
        """
        Initialize PestProject as a component of Calibrator.
        
        Args:
            calibrator: Parent Calibrator object
        """
        self.calibrator = calibrator
        self.project_path: Optional[Path] = None
        
        # In-memory storage
        self._observations: List[Dict[str, Any]] = []
        self._parameters: List[Dict[str, Any]] = []
        self._secondary_objectives: List[Dict[str, Any]] = []
        
        # Lookup dictionaries for quick access by name
        self._obs_lookup: Dict[str, int] = {}  # name -> index in _observations
        self._par_lookup: Dict[str, int] = {}  # name -> index in _parameters
        self._sec_obj_lookup: Dict[str, int] = {}  # name -> index in _secondary_objectives
    
    # -------------------------------------------------------------------------
    # Project Setup
    # -------------------------------------------------------------------------
    
    def create_project(self, folder_name: str = 'pest') -> Path:
        """
        Create a PEST project directory and copy model files.
        
        Creates a 'pest' subdirectory within the calibrator's project path
        and copies the original HSPF model files into it.
        
        Args:
            folder_name: Name of the pest project folder (default: 'pest')
            
        Returns:
            Path to the created project directory
        """
        # Create pest folder within calibrator project
        self.project_path = Path(self.calibrator.project_path) / folder_name
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to pest directory
        self._copy_model_files()
        
        return self.project_path
    
    def _copy_model_files(self) -> None:
        """Copy original HSPF model files to pest project directory."""
        if self.project_path is None:
            raise RuntimeError("Project not created. Call create_project() first.")
        
        # Copy UCI file
        src_uci = Path(self.calibrator.uci.filepath)
        dst_uci = self.project_path / src_uci.name
        shutil.copy2(src_uci, dst_uci)
        
        # Copy WDM file if it exists
        if hasattr(self.calibrator, 'wdm') and self.calibrator.wdm:
            src_wdm = Path(self.calibrator.wdm.filepath)
            if src_wdm.exists():
                dst_wdm = self.project_path / src_wdm.name
                shutil.copy2(src_wdm, dst_wdm)
        
        # Copy HDF5 output file if it exists
        if hasattr(self.calibrator, 'hdf5_path') and self.calibrator.hdf5_path:
            src_hdf = Path(self.calibrator.hdf5_path)
            if src_hdf.exists():
                dst_hdf = self.project_path / src_hdf.name
                shutil.copy2(src_hdf, dst_hdf)
    
    # -------------------------------------------------------------------------
    # Observation Methods
    # -------------------------------------------------------------------------
    
    def add_observation(
        self,
        name: str,
        station_id: str,
        reach_ids: Union[int, List[int]],
        constituent: str,
        time_step: str = 'daily',
        weight: float = 1.0,
        obgnme: Optional[str] = None,
    ) -> None:
        """
        Add an observation location for PEST calibration.
        
        Observations define which station data to compare model output against.
        Each observation represents a measurement location (station) and the
        model reaches that contribute to that location.
        
        Args:
            name: Unique identifier for this observation (e.g., 'st1_flow')
            station_id: ID of the monitoring station in the station data
            reach_ids: HSPF reach ID(s) that contribute to this station
            constituent: What is being measured ('flow', 'sediment', etc.)
            time_step: Temporal resolution ('daily', 'hourly', 'monthly')
            weight: Initial observation weight for PEST (default: 1.0)
            obgnme: Observation group name (default: constituent)
            
        Raises:
            ValueError: If observation name already exists
        """
        if name in self._obs_lookup:
            raise ValueError(f"Observation '{name}' already exists. Use adjust_observation() to modify.")
        
        # Ensure reach_ids is a list
        if isinstance(reach_ids, int):
            reach_ids = [reach_ids]
        
        obs_config = {
            'name': name,
            'station_id': station_id,
            'reach_ids': reach_ids,
            'constituent': constituent,
            'time_step': time_step,
            'weight': weight,
            'obgnme': obgnme or constituent,
        }
        
        self._observations.append(obs_config)
        self._obs_lookup[name] = len(self._observations) - 1
    
    def adjust_observation(self, name: str, **kwargs) -> None:
        """
        Adjust properties of an existing observation.
        
        Args:
            name: Name of the observation to adjust
            **kwargs: Properties to update (weight, obgnme, etc.)
            
        Raises:
            KeyError: If observation name not found
        """
        if name not in self._obs_lookup:
            raise KeyError(f"Observation '{name}' not found.")
        
        idx = self._obs_lookup[name]
        for key, value in kwargs.items():
            if key in self._observations[idx]:
                self._observations[idx][key] = value
            else:
                raise KeyError(f"Unknown observation property: {key}")
    
    def get_observation(self, name: str) -> Dict[str, Any]:
        """Get observation configuration by name."""
        if name not in self._obs_lookup:
            raise KeyError(f"Observation '{name}' not found.")
        return self._observations[self._obs_lookup[name]].copy()
    
    def list_observations(self) -> List[str]:
        """List all observation names."""
        return list(self._obs_lookup.keys())
    
    # -------------------------------------------------------------------------
    # Secondary Objectives Methods
    # -------------------------------------------------------------------------
    
    def add_secondary_objective(
        self,
        name: str,
        target_value: float,
        weight: float = 1.0,
        obgnme: str = 'secondary',
        description: str = ''
    ) -> None:
        """
        Add a secondary objective (target/constraint) independent of observation data.
        
        Secondary objectives are treated as observations in PEST but represent
        targets or constraints rather than time series data. Examples:
        - Annual average runoff should be X inches
        - Summer loading rate should be Y lbs/acre
        
        Args:
            name: Unique name for this objective (will be used as obsname)
            target_value: The target value to achieve
            weight: PEST weight for this objective
            obgnme: Observation group name (default: 'secondary')
            description: Optional description of the objective
            
        Raises:
            ValueError: If name already exists
        """
        if name in self._sec_obj_lookup:
            raise ValueError(f"Secondary objective '{name}' already exists.")
        
        sec_obj = {
            'name': name,
            'target_value': target_value,
            'weight': weight,
            'obgnme': obgnme,
            'description': description,
        }
        
        self._secondary_objectives.append(sec_obj)
        self._sec_obj_lookup[name] = len(self._secondary_objectives) - 1
    
    def adjust_secondary_objective(self, name: str, **kwargs) -> None:
        """
        Adjust properties of an existing secondary objective.
        
        Args:
            name: Name of the objective to modify
            **kwargs: Properties to update (target_value, weight, obgnme, description)
            
        Raises:
            KeyError: If objective not found
        """
        if name not in self._sec_obj_lookup:
            raise KeyError(f"Secondary objective '{name}' not found.")
        
        idx = self._sec_obj_lookup[name]
        for key, value in kwargs.items():
            if key in self._secondary_objectives[idx]:
                self._secondary_objectives[idx][key] = value
    
    def list_secondary_objectives(self) -> List[str]:
        """List all secondary objective names."""
        return list(self._sec_obj_lookup.keys())
    
    # -------------------------------------------------------------------------
    # Parameter Methods
    # -------------------------------------------------------------------------
    
    def add_parameter(
        self,
        block: str,
        table: str,
        table_id: int,
        parameter: str,
        single_template: bool = True,
        group: Optional[str] = None,
        opnids: Optional[List[int]] = None,
        **kwargs
    ) -> str:
        """
        Add an HSPF parameter for PEST optimization.
        
        Parameters define which HSPF model values will be adjusted by PEST.
        Uses HSPF_PARAMETER_DEFAULTS for common parameters; custom bounds
        can be provided via kwargs.
        
        Args:
            block: UCI block name (e.g., 'PERLND', 'IMPLND', 'RCHRES')
            table: UCI table name (e.g., 'PWAT-PARM2', 'PWAT-PARM4')
            table_id: Table index (usually 0 for hydrology)
            parameter: Parameter name (e.g., 'LZSN', 'INFILT')
            single_template: If True, one parameter for all operations.
                           If False, separate parameter per operation ID.
            group: Character to append for grouping (e.g., 'A', 'B')
            opnids: Specific operation IDs to include (None = all)
            **kwargs: Override default parameter properties:
                      parval1, parlbnd, parubnd, pargp, partrans, parchglim
        
        Returns:
            Parameter name(s) that will be created
            
        Raises:
            ValueError: If parameter name already exists
        """
        # Build base parameter name
        parnme = parameter.lower()
        if group:
            parnme = f"{parnme}{group}"
        
        if parnme in self._par_lookup and single_template:
            raise ValueError(f"Parameter '{parnme}' already exists. Use adjust_parameter() to modify.")
        
        # Get defaults if available (using tuple key)
        defaults_key = (block, table, table_id, parameter)
        defaults = HSPF_PARAMETER_DEFAULTS.get(defaults_key, {})
        
        par_config = {
            'parnme': parnme,
            'block': block,
            'table': table,
            'table_id': table_id,
            'parameter': parameter,
            'single_template': single_template,
            'group': group,
            'opnids': opnids,
            # PEST properties with defaults
            'parval1': kwargs.get('parval1', defaults.get('parval1', 1.0)),
            'parlbnd': kwargs.get('parlbnd', defaults.get('parlbnd', 0.0)),
            'parubnd': kwargs.get('parubnd', defaults.get('parubnd', 10.0)),
            'pargp': kwargs.get('pargp', defaults.get('pargp', 'default')),
            'partrans': kwargs.get('partrans', 'none'),
            'parchglim': kwargs.get('parchglim', 'relative'),
            'scale': kwargs.get('scale', 1.0),
            'offset': kwargs.get('offset', 0.0),
        }
        
        self._parameters.append(par_config)
        self._par_lookup[parnme] = len(self._parameters) - 1
        
        return parnme
    
    def adjust_parameter(self, name: str, **kwargs) -> None:
        """
        Adjust properties of an existing parameter.
        
        Args:
            name: Parameter name to adjust
            **kwargs: Properties to update (parval1, parlbnd, parubnd, pargp, etc.)
            
        Raises:
            KeyError: If parameter name not found
        """
        if name not in self._par_lookup:
            raise KeyError(f"Parameter '{name}' not found.")
        
        idx = self._par_lookup[name]
        for key, value in kwargs.items():
            if key in self._parameters[idx]:
                self._parameters[idx][key] = value
            else:
                raise KeyError(f"Unknown parameter property: {key}")
    
    def get_parameter(self, name: str) -> Dict[str, Any]:
        """Get parameter configuration by name."""
        if name not in self._par_lookup:
            raise KeyError(f"Parameter '{name}' not found.")
        return self._parameters[self._par_lookup[name]].copy()
    
    def list_parameters(self) -> List[str]:
        """List all parameter names."""
        return list(self._par_lookup.keys())
    
    # -------------------------------------------------------------------------
    # Build Methods
    # -------------------------------------------------------------------------
    
    def build_parameters(self) -> pd.DataFrame:
        """
        Build a parameter DataFrame for pyemu.Pst.
        
        Returns:
            DataFrame with columns matching pyemu parameter_data format:
            parnme, partrans, parchglim, parval1, parlbnd, parubnd, pargp, scale, offset
        """
        if not self._parameters:
            return pd.DataFrame()
        
        records = []
        for par in self._parameters:
            records.append({
                'parnme': par['parnme'],
                'partrans': par['partrans'],
                'parchglim': par['parchglim'],
                'parval1': par['parval1'],
                'parlbnd': par['parlbnd'],
                'parubnd': par['parubnd'],
                'pargp': par['pargp'],
                'scale': par['scale'],
                'offset': par['offset'],
            })
        
        df = pd.DataFrame(records)
        df.set_index('parnme', inplace=True)
        return df
    
    def build_observations(self) -> pd.DataFrame:
        """
        Build an observation DataFrame for pyemu.Pst.
        
        Note: This returns observation *configurations*, not actual observation
        values. Use build_observation_data() to create the full observation
        dataset with values from station data.
        
        This includes both regular observations and secondary objectives.
        
        Returns:
            DataFrame with observation configurations:
            name, station_id, reach_ids, constituent, time_step, weight, obgnme, type
        """
        records = []
        
        # Add regular observations
        for obs in self._observations:
            record = obs.copy()
            record['type'] = 'timeseries'
            records.append(record)
        
        # Add secondary objectives
        for sec_obj in self._secondary_objectives:
            records.append({
                'name': sec_obj['name'],
                'station_id': None,
                'reach_ids': None,
                'constituent': None,
                'time_step': None,
                'weight': sec_obj['weight'],
                'obgnme': sec_obj['obgnme'],
                'target_value': sec_obj['target_value'],
                'description': sec_obj['description'],
                'type': 'secondary',
            })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df.set_index('name', inplace=True)
        return df
    
    def build_observation_data(self) -> pd.DataFrame:
        """
        Build full observation data using calibrator's compare_simulated_observed.
        
        This method fetches aligned observed/simulated timeseries for each
        observation configuration and creates PEST-ready observation records.
        Rows with missing observed data are dropped.
        
        Returns:
            DataFrame with columns: obsname, obsval, weight, obgnme
            Ready for use with pyemu.Pst
            
        Raises:
            RuntimeError: If no observations have been added
        """
        if not self._observations and not self._secondary_objectives:
            raise RuntimeError("No observations or secondary objectives added. Use add_observation() first.")
        
        all_records = []
        
        # Process each timeseries observation
        for obs in self._observations:
            # Get aligned observed/simulated data from calibrator
            df = self.calibrator.compare_simulated_observed(
                station_ids=obs['station_id'],
                reach_ids=obs['reach_ids'],
                constituent=obs['constituent'],
                time_step=obs['time_step']
            )
            
            # Drop rows with missing observed values
            if 'observed' in df.columns:
                df = df.dropna(subset=['observed'])
            elif 'observed_flow' in df.columns:
                df = df.dropna(subset=['observed_flow'])
            
            # Create observation records
            for i, (idx, row) in enumerate(df.iterrows()):
                # Determine which observed column to use
                if 'observed' in df.columns:
                    obsval = row['observed']
                elif 'observed_flow' in df.columns:
                    obsval = row['observed_flow']
                else:
                    continue
                
                # Create unique observation name
                date_str = idx.strftime('%Y%m%d') if hasattr(idx, 'strftime') else str(i)
                obsname = f"{obs['name']}_{date_str}"
                
                all_records.append({
                    'obsname': obsname,
                    'obsval': obsval,
                    'weight': obs['weight'],
                    'obgnme': obs['obgnme'],
                })
        
        # Add secondary objectives
        for sec_obj in self._secondary_objectives:
            all_records.append({
                'obsname': sec_obj['name'],
                'obsval': sec_obj['target_value'],
                'weight': sec_obj['weight'],
                'obgnme': sec_obj['obgnme'],
            })
        
        if not all_records:
            return pd.DataFrame(columns=['obsname', 'obsval', 'weight', 'obgnme'])
        
        df = pd.DataFrame(all_records)
        df.set_index('obsname', inplace=True)
        return df
    
    # -------------------------------------------------------------------------
    # Summary Methods
    # -------------------------------------------------------------------------
    
    def summary(self) -> str:
        """Return a summary of the PEST project configuration."""
        lines = [
            "PEST Project Summary",
            "=" * 40,
            f"Project Path: {self.project_path}",
            f"Observations: {len(self._observations)}",
        ]
        
        for obs in self._observations:
            lines.append(f"  - {obs['name']}: {obs['constituent']} @ station {obs['station_id']} (weight={obs['weight']})")
        
        lines.append(f"Secondary Objectives: {len(self._secondary_objectives)}")
        for sec_obj in self._secondary_objectives:
            lines.append(f"  - {sec_obj['name']}: target={sec_obj['target_value']} (weight={sec_obj['weight']})")
        
        lines.append(f"Parameters: {len(self._parameters)}")
        for par in self._parameters:
            lines.append(f"  - {par['parnme']}: {par['parval1']} [{par['parlbnd']}, {par['parubnd']}]")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"PestProject(observations={len(self._observations)}, secondary_objectives={len(self._secondary_objectives)}, parameters={len(self._parameters)})"
