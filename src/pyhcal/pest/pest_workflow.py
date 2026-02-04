# -*- coding: utf-8 -*-
"""
PEST++ Workflow Module for HSPF Model Calibration.

This module provides a thin coordinator class that delegates to specialized
modules for each concern:

- parameter_manager.py: Parameter storage and HSPF defaults
- observation_manager.py: Observation storage and data management  
- observation_builder.py: Building PEST observations from timeseries
- file_manager.py: Template/instruction files, forward run scripts
- metrics.py: Statistical calculations

Uses pyemu directly for PST file creation (no unnecessary wrapper).

Usage:
    from pyhcal.pest_workflow import PestWorkflow
    
    pest = PestWorkflow("my_project")
    pest.load_model(0)
    pest.add_hspf_parameter('PERLND', 'PWAT-PARM2', 0, 'LZSN')
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
from .file_manager import FileManager
from pyhcal.calibrators import calibrator

class PestWorkflow:
    """
    Thin coordinator for PEST++ workflows.
    
    Delegates to specialized managers:
    - self.parameters: ParameterManager
    - self.observations: ObservationManager  
    - self.files: FileManager
    
    Uses pyemu.Pst directly for PST file creation.
    
    Example:
        pest = PestWorkflow("my_project")
        pest.load_model(0)
        pest.add_hspf_parameter('PERLND', 'PWAT-PARM2', 0, 'LZSN')
        pest.add_observation("outlet", "flow", observed=obs, simulated=sim)
        pest.build()
        pest.write("my_case.pst")
    """
    
    def __init__(
        self, 
        project_path: Union[str, Path],
        build_settings: Optional[BuildSettings] = None,
        pest_config: Optional[PestConfig] = None
    ):
        """
        Initialize the PEST workflow.
        
        Args:
            project_path: Path to project folder (pest/ subdirectory will be created)
            build_settings: Settings for observation building
            pest_config: PEST++ configuration
        """
        if pyemu is None:
            raise ImportError("pyemu is required. Install with: pip install pyemu")
        
        self.cal = calibrator(project_path)
        self.project_path = Path(project_path)
        self.pest_path = self.project_path / 'pest'
        self.uci = None
        
        # Initialize managers
        self.parameters = ParameterManager()
        self.observations = ObservationManager(build_settings)
        self.files = FileManager(self.pest_path)
        
        # Configuration
        self._config = pest_config or PestConfig()
        
        # State - use pyemu.Pst directly
        self._pst: Optional[pyemu.Pst] = None
    
    # =========================================================================
    # Parameter Methods (delegate to ParameterManager)
    # =========================================================================
    
    def load_model(self,name):
        self.cal.load_model(name)
        self.uci = self.cal.uci

    def add_hspf_parameter(self, block, table, table_id, parameter,single_template=True, template_group='',opnids = None):
        pest_defaults = self.parameters.get_hspf_defaults(block, table,table_id,parameter)

        pest_params = self.cal.uci.add_parameter_template(block,
                                            table,
                                            table_id,
                                            parameter,
                                            single_template=single_template,
                                            group_id=template_group,
                                            opnids = opnids)
        for param in pest_params:
            self.parameters.add(name = param, 
                                initial_value=pest_defaults['initial_value'],
                                lower_bound=pest_defaults['lower_bound'], 
                                upper_bound=pest_defaults['upper_bound'],
                                transform=pest_defaults['transform'])


        return self
    
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
        Build the complete PEST setup using pyemu directly.
        
        This:
        1. Builds observation records from data
        2. Creates pyemu.Pst directly from parameter/observation names
        3. Sets parameter and observation data
        
        Returns:
            pyemu.Pst object
        """
        # Build observation records
        obs_data = self.observations.build_observations()
        par_data = self.parameters.to_dataframe()
        
        # Get names for pyemu
        par_names = list(par_data.index) if not par_data.empty else []
        obs_names = list(obs_data["obsnme"]) if not obs_data.empty else []
        
        # Create PST directly using pyemu
        self._pst = pyemu.Pst.from_par_obs_names(
            par_names=par_names,
            obs_names=obs_names
        )
        
        # Set parameter data
        if not par_data.empty:
            for pname in par_data.index:
                for col in ["parval1", "parlbnd", "parubnd", "pargp", "partrans"]:
                    if col in par_data.columns:
                        self._pst.parameter_data.loc[pname, col] = par_data.loc[pname, col]
        
        # Set observation data
        if not obs_data.empty:
            for _, row in obs_data.iterrows():
                oname = row["obsnme"]
                if oname in self._pst.observation_data.index:
                    self._pst.observation_data.loc[oname, "obsval"] = row.get("obsval", 0.0)
                    self._pst.observation_data.loc[oname, "weight"] = row.get("weight", 1.0)
                    self._pst.observation_data.loc[oname, "obgnme"] = row.get("obgnme", "obsgroup")
        
        # Apply PEST++ configuration
        self._pst.control_data.noptmax = self._config.noptmax
        self._pst.svd_data.maxsing = self._config.svd_maxsing
        self._pst.svd_data.eigthresh = self._config.svd_eigthresh
        if self._config.pestpp_options:
            for key, value in self._config.pestpp_options.items():
                self._pst.pestpp_options[key] = value
        
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
        model_command: str = "python forward_run.py",
        create_template: bool = True
    ) -> Path:
        """
        Write all PEST files.
        
        Args:
            filename: Name of PST file
            model_command: Command to run forward model
            create_template: If True and UCI is available, create template file
            
        Returns:
            Path to PST file
        """
        if self._pst is None:
            raise ValueError("Call build() first")
        
        # Get observation names
        obs_data = self.observations.get_built_observations()
        obs_names = list(obs_data["obsnme"]) if obs_data is not None else []
        
        # Set model command directly on pyemu.Pst
        self._pst.model_command = [model_command]
        
        # Create UCI template file if UCI available
        if create_template and self.uci is not None:
            tpl_path = self.files.pest_dir / "model.tpl"
            uci_path = self.files.pest_dir / "model.uci"
            self.cal.uci.write_tpl(new_tpl_path=tpl_path)
            
            # Add template file to PST
            self._add_template_file(tpl_path, uci_path)
        
        # Write instruction file
        ins_path = self.files.write_instruction_file(obs_names)
        out_path = self.files.pest_dir / "model_output.out"
        self._add_instruction_file(ins_path, out_path)
        
        # Write output template
        self.files.write_output_file_template(obs_names)
        
        # Write forward run script
        self.files.write_forward_run_script(
            model_command=model_command,
            observation_names=obs_names
        )
        
        # Write PST file using pyemu directly
        pst_path = self.files.pest_dir / filename
        self._pst.write(str(pst_path))
        
        return pst_path
    
    def _add_template_file(self, template_path: Path, model_input_path: Path) -> None:
        """Add a template file to the PST."""
        new_row = pd.DataFrame({
            "pest_file": [str(template_path)],
            "model_file": [str(model_input_path)]
        })
        self._pst.model_input_data = pd.concat(
            [self._pst.model_input_data, new_row],
            ignore_index=True
        )
    
    def _add_instruction_file(self, instruction_path: Path, model_output_path: Path) -> None:
        """Add an instruction file to the PST."""
        new_row = pd.DataFrame({
            "pest_file": [str(instruction_path)],
            "model_file": [str(model_output_path)]
        })
        self._pst.model_output_data = pd.concat(
            [self._pst.model_output_data, new_row],
            ignore_index=True
        )
    
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
