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
HSPF_PARAMETER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # PERLND parameters - PWAT-PARM2 (table_id=0)
    'LZSN':   {'parval1': 6.0,  'parlbnd': 2.0,   'parubnd': 15.0,  'pargp': 'storage'},
    'INFILT': {'parval1': 0.10, 'parlbnd': 0.001, 'parubnd': 0.50,  'pargp': 'infiltration'},
    'LSUR':   {'parval1': 400,  'parlbnd': 100,   'parubnd': 700,   'pargp': 'surface'},
    'SLSUR':  {'parval1': 0.10, 'parlbnd': 0.001, 'parubnd': 0.30,  'pargp': 'surface'},
    'KVARY':  {'parval1': 0.0,  'parlbnd': 0.0,   'parubnd': 5.0,   'pargp': 'groundwater'},
    'AGWRC':  {'parval1': 0.98, 'parlbnd': 0.85,  'parubnd': 0.999, 'pargp': 'groundwater'},
    
    # PERLND parameters - PWAT-PARM3 (table_id=0)
    'PETMAX': {'parval1': 40.0, 'parlbnd': 32.0,  'parubnd': 48.0,  'pargp': 'et'},
    'PETMIN': {'parval1': 35.0, 'parlbnd': 30.0,  'parubnd': 40.0,  'pargp': 'et'},
    'INFEXP': {'parval1': 2.0,  'parlbnd': 1.0,   'parubnd': 10.0,  'pargp': 'infiltration'},
    'INFILD': {'parval1': 2.0,  'parlbnd': 1.0,   'parubnd': 2.0,   'pargp': 'infiltration'},
    'DEEPFR': {'parval1': 0.0,  'parlbnd': 0.0,   'parubnd': 0.50,  'pargp': 'groundwater'},
    'BASETP': {'parval1': 0.0,  'parlbnd': 0.0,   'parubnd': 0.20,  'pargp': 'et'},
    'AGWETP': {'parval1': 0.0,  'parlbnd': 0.0,   'parubnd': 0.20,  'pargp': 'et'},
    
    # PERLND parameters - PWAT-PARM4 (table_id=0)
    'CEPSC':  {'parval1': 0.1,  'parlbnd': 0.01,  'parubnd': 0.40,  'pargp': 'interception'},
    'UZSN':   {'parval1': 1.0,  'parlbnd': 0.05,  'parubnd': 5.0,   'pargp': 'storage'},
    'NSUR':   {'parval1': 0.25, 'parlbnd': 0.05,  'parubnd': 0.50,  'pargp': 'surface'},
    'INTFW':  {'parval1': 1.5,  'parlbnd': 0.0,   'parubnd': 10.0,  'pargp': 'interflow'},
    'IRC':    {'parval1': 0.70, 'parlbnd': 0.30,  'parubnd': 0.85,  'pargp': 'interflow'},
    'LZETP':  {'parval1': 0.30, 'parlbnd': 0.0,   'parubnd': 0.90,  'pargp': 'et'},
    
    # IMPLND parameters - IWAT-PARM2 (table_id=0)
    'LSUR_I': {'parval1': 400,  'parlbnd': 100,   'parubnd': 700,   'pargp': 'surface'},
    'SLSUR_I':{'parval1': 0.10, 'parlbnd': 0.001, 'parubnd': 0.30,  'pargp': 'surface'},
    'NSUR_I': {'parval1': 0.10, 'parlbnd': 0.01,  'parubnd': 0.25,  'pargp': 'surface'},
    'RETSC':  {'parval1': 0.10, 'parlbnd': 0.01,  'parubnd': 0.30,  'pargp': 'interception'},
    
    # RCHRES parameters - HYDR-PARM2 (table_id=0)
    'FTBUCI': {'parval1': 1.0,  'parlbnd': 0.5,   'parubnd': 2.0,   'pargp': 'routing'},
    'KS':     {'parval1': 0.5,  'parlbnd': 0.0,   'parubnd': 1.0,   'pargp': 'routing'},
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
        
        # Lookup dictionaries for quick access by name
        self._obs_lookup: Dict[str, int] = {}  # name -> index in _observations
        self._par_lookup: Dict[str, int] = {}  # name -> index in _parameters
    
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
        
        # Get defaults if available
        defaults = HSPF_PARAMETER_DEFAULTS.get(parameter, {})
        
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
        
        Returns:
            DataFrame with observation configurations:
            name, station_id, reach_ids, constituent, time_step, weight, obgnme
        """
        if not self._observations:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._observations)
        df.set_index('name', inplace=True)
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
        
        lines.append(f"Parameters: {len(self._parameters)}")
        for par in self._parameters:
            lines.append(f"  - {par['parnme']}: {par['parval1']} [{par['parlbnd']}, {par['parubnd']}]")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"PestProject(observations={len(self._observations)}, parameters={len(self._parameters)})"
