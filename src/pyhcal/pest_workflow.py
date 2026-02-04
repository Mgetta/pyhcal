# -*- coding: utf-8 -*-
"""
PEST++ Workflow Module for HSPF Model Calibration.

This module provides a thin coordinator class that delegates to specialized
modules for each concern:

- parameter_manager.py: Parameter storage and HSPF defaults
- observation_manager.py: Observation storage and data management  
- observation_builder.py: Building PEST observations from timeseries
- pst_builder.py: Creating pyemu PST files
- file_manager.py: Template/instruction files, forward run scripts
- metrics.py: Statistical calculations

Usage:
    from pyhcal.pest_workflow import PestWorkflow
    
    pest = PestWorkflow("my_project")
    pest.add_hydrology_defaults()
    pest.add_observation("outlet", "flow", observed=obs, simulated=sim)
    pest.build()
    pest.write("my_case.pst")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

try:
    import pyemu
except ImportError:
    pyemu = None

from .pest_types import Parameter, ObservationConfig, BuildSettings, PestConfig
from .parameter_manager import ParameterManager, get_hspf_hydrology_defaults
from .observation_manager import ObservationManager
from .pst_builder import PstBuilder
from .file_manager import FileManager


class PestWorkflow:
    """
    Thin coordinator for PEST++ workflows.
    
    Delegates to specialized managers:
    - self.parameters: ParameterManager
    - self.observations: ObservationManager  
    - self.pst_builder: PstBuilder
    - self.files: FileManager
    
    Example:
        pest = PestWorkflow("my_project")
        pest.add_hydrology_defaults()
        pest.add_observation("outlet", "flow", observed=obs, simulated=sim)
        pest.build()
        pest.write("my_case.pst")
    """
    
    def __init__(
        self, 
        project_path: Union[str, Path],
        uci=None,
        build_settings: Optional[BuildSettings] = None,
        pest_config: Optional[PestConfig] = None
    ):
        """
        Initialize the PEST workflow.
        
        Args:
            project_path: Path to project folder (pest/ subdirectory will be created)
            uci: Optional UCI object from hspf package
            build_settings: Settings for observation building
            pest_config: PEST++ configuration
        """
        if pyemu is None:
            raise ImportError("pyemu is required. Install with: pip install pyemu")
        
        self.project_path = Path(project_path)
        self.pest_path = self.project_path / 'pest'
        self.uci = uci
        
        # Initialize managers
        self.parameters = ParameterManager()
        self.observations = ObservationManager(build_settings)
        self.pst_builder = PstBuilder(pest_config)
        self.files = FileManager(self.pest_path)
        
        # State
        self._pst: Optional[pyemu.Pst] = None
    
    # =========================================================================
    # Parameter Methods (delegate to ParameterManager)
    # =========================================================================
    
    def add_parameter(
        self,
        name: str,
        initial_value: float,
        lower_bound: float,
        upper_bound: float,
        transform: str = "none",
        group: Optional[str] = None,
        **metadata
    ) -> "PestWorkflow":
        """
        Add a parameter.
        
        Args:
            name: Parameter name (max 12 chars)
            initial_value: Starting value
            lower_bound: Minimum value
            upper_bound: Maximum value
            transform: "none", "log", or "fixed"
            group: Parameter group
            **metadata: Additional metadata (block, table, etc.)
            
        Returns:
            Self for method chaining
        """
        self.parameters.add(
            name=name,
            initial_value=initial_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=transform,
            group=group,
            **metadata
        )
        return self
    
    def add_hydrology_defaults(
        self, 
        only: Optional[List[str]] = None
    ) -> "PestWorkflow":
        """
        Add default HSPF hydrology parameters.
        
        Args:
            only: If provided, only add these parameters
            
        Returns:
            Self for method chaining
        """
        defaults = get_hspf_hydrology_defaults()
        self.parameters.add_from_defaults(defaults, only=only)
        return self
    
    # =========================================================================
    # Observation Methods (delegate to ObservationManager)
    # =========================================================================
    
    def add_observation(
        self,
        name: str,
        group: str,
        obs_type: str = "flow",
        weight: float = 1.0,
        observed: Optional[pd.Series] = None,
        simulated: Optional[pd.Series] = None,
        **metadata
    ) -> "PestWorkflow":
        """
        Add an observation location.
        
        Args:
            name: Observation name
            group: PEST observation group
            obs_type: Type (flow, sediment, etc.)
            weight: Weight in objective function
            observed: Observed timeseries data
            simulated: Simulated timeseries data
            **metadata: Additional metadata
            
        Returns:
            Self for method chaining
        """
        self.observations.add(
            name=name,
            group=group,
            obs_type=obs_type,
            weight=weight,
            observed=observed,
            simulated=simulated,
            **metadata
        )
        return self
    
    def set_observation_data(
        self,
        name: str,
        observed: Optional[pd.Series] = None,
        simulated: Optional[pd.Series] = None
    ) -> "PestWorkflow":
        """
        Set data for an existing observation.
        
        Args:
            name: Observation name
            observed: Observed timeseries
            simulated: Simulated timeseries
            
        Returns:
            Self for method chaining
        """
        self.observations.set_data(name, observed, simulated)
        return self
    
    # =========================================================================
    # Build Methods
    # =========================================================================
    
    def build(self) -> "pyemu.Pst":
        """
        Build the complete PEST setup.
        
        This:
        1. Builds observation records from data
        2. Creates the PST file
        
        Returns:
            pyemu.Pst object
        """
        # Build observation records
        obs_data = self.observations.build_observations()
        
        # Build PST
        self._pst = self.pst_builder.build(
            parameter_data=self.parameters.to_dataframe(),
            observation_data=obs_data
        )
        
        return self._pst
    
    def build_observations(
        self,
        settings: Optional[BuildSettings] = None
    ) -> pd.DataFrame:
        """
        Build PEST observations from data.
        
        Args:
            settings: Override default build settings
            
        Returns:
            DataFrame of observation records
        """
        return self.observations.build_observations(settings)
    
    # =========================================================================
    # File Methods (delegate to FileManager)
    # =========================================================================
    
    def write(
        self,
        filename: str = "pest.pst",
        model_command: str = "python forward_run.py"
    ) -> Path:
        """
        Write all PEST files.
        
        Args:
            filename: Name of PST file
            model_command: Command to run forward model
            
        Returns:
            Path to PST file
        """
        if self._pst is None:
            raise ValueError("Call build() first")
        
        # Get observation names
        obs_data = self.observations.get_built_observations()
        obs_names = list(obs_data["obsnme"]) if obs_data is not None else []
        
        # Set model command
        self.pst_builder.set_model_command(model_command)
        
        # Write instruction file
        ins_path = self.files.write_instruction_file(obs_names)
        out_path = self.files.pest_dir / "model_output.out"
        self.pst_builder.add_instruction_file(ins_path, out_path)
        
        # Write output template
        self.files.write_output_file_template(obs_names)
        
        # Write forward run script
        self.files.write_forward_run_script(
            model_command=model_command,
            observation_names=obs_names
        )
        
        # Write PST file
        pst_path = self.files.pest_dir / filename
        self.pst_builder.write(pst_path)
        
        return pst_path
    
    def save_observed_data(self, filename: str = "observed_data.csv") -> Path:
        """
        Save observed data for forward run script.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        # Collect observed data from all observations
        data = {}
        for name in self.observations.names:
            obs_data = self.observations.get_data(name)
            if "observed" in obs_data:
                data[name] = obs_data["observed"]
        
        return self.files.save_observed_data(data, filename)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def pst(self) -> Optional["pyemu.Pst"]:
        """Get the built PST object."""
        return self._pst
    
    # =========================================================================
    # Summary Methods
    # =========================================================================
    
    def get_parameter_summary(self) -> pd.DataFrame:
        """Get DataFrame summary of parameters."""
        return self.parameters.to_dataframe()
    
    def get_observation_summary(self) -> pd.DataFrame:
        """Get DataFrame summary of observation configurations."""
        records = []
        for config in self.observations:
            records.append({
                "name": config.name,
                "group": config.group,
                "obs_type": config.obs_type,
                "weight": config.weight,
                "has_data": self.observations.has_data(config.name)
            })
        return pd.DataFrame(records)


# Convenience function for quick setup
def create_pest_workflow(
    project_path: Union[str, Path],
    uci=None,
    **kwargs
) -> PestWorkflow:
    """Create a PestWorkflow instance."""
    return PestWorkflow(project_path, uci, **kwargs)
