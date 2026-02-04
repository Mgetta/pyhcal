"""
PEST Control File (PST) builder using pyemu.

This module handles:
- Creating pyemu Pst objects from parameters and observations
- Configuring PEST++ settings
- Writing PST files
"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd

try:
    import pyemu
except ImportError:
    pyemu = None

from .pest_types import PestConfig


class PstBuilder:
    """Builds PEST control files using pyemu."""
    
    def __init__(self, config: Optional[PestConfig] = None):
        """
        Initialize PST builder.
        
        Args:
            config: PEST configuration settings
        """
        if pyemu is None:
            raise ImportError("pyemu is required but not installed. Install with: pip install pyemu")
        
        self._config = config or PestConfig()
        self._pst: Optional[pyemu.Pst] = None
    
    def build(
        self,
        parameter_data: pd.DataFrame,
        observation_data: pd.DataFrame
    ) -> "pyemu.Pst":
        """
        Build a Pst object from parameter and observation data.
        
        Args:
            parameter_data: DataFrame with parameter info (from ParameterManager.to_dataframe())
            observation_data: DataFrame with observation info (from ObservationManager.build_observations())
            
        Returns:
            pyemu.Pst object
        """
        # Create empty Pst
        self._pst = pyemu.Pst.from_par_obs_names(
            par_names=list(parameter_data.index) if not parameter_data.empty else [],
            obs_names=list(observation_data["obsnme"]) if not observation_data.empty else []
        )
        
        # Set parameter data
        if not parameter_data.empty:
            self._set_parameter_data(parameter_data)
        
        # Set observation data
        if not observation_data.empty:
            self._set_observation_data(observation_data)
        
        # Apply PEST++ configuration
        self._apply_config()
        
        return self._pst
    
    def write(self, filepath: Union[str, Path]) -> None:
        """
        Write PST file to disk.
        
        Args:
            filepath: Path to write PST file
        """
        if self._pst is None:
            raise ValueError("No Pst built yet. Call build() first.")
        
        self._pst.write(str(filepath))
    
    def get_pst(self) -> Optional["pyemu.Pst"]:
        """Get the built Pst object."""
        return self._pst
    
    def _set_parameter_data(self, parameter_data: pd.DataFrame) -> None:
        """Set parameter data on the Pst object."""
        par_df = self._pst.parameter_data
        
        for pname in parameter_data.index:
            if pname in par_df.index:
                for col in ["parval1", "parlbnd", "parubnd", "pargp", "partrans", "parchglim", "scale", "offset"]:
                    if col in parameter_data.columns:
                        par_df.loc[pname, col] = parameter_data.loc[pname, col]
    
    def _set_observation_data(self, observation_data: pd.DataFrame) -> None:
        """Set observation data on the Pst object."""
        obs_df = self._pst.observation_data
        
        # Map our column names to pyemu column names
        column_map = {
            "obsnme": "obsnme",
            "obsval": "obsval", 
            "weight": "weight",
            "obgnme": "obgnme"
        }
        
        for _, row in observation_data.iterrows():
            oname = row["obsnme"]
            if oname in obs_df.index:
                obs_df.loc[oname, "obsval"] = row.get("obsval", 0.0)
                obs_df.loc[oname, "weight"] = row.get("weight", 1.0)
                obs_df.loc[oname, "obgnme"] = row.get("obgnme", "obsgroup")
    
    def _apply_config(self) -> None:
        """Apply PEST++ configuration settings."""
        config = self._config
        
        # Control data settings
        self._pst.control_data.noptmax = config.noptmax
        
        # SVD settings
        self._pst.svd_data.maxsing = config.svd_maxsing
        self._pst.svd_data.eigthresh = config.svd_eigthresh
        
        # PEST++ options
        if config.pestpp_options:
            for key, value in config.pestpp_options.items():
                self._pst.pestpp_options[key] = value
    
    def set_model_command(self, command: str) -> "PstBuilder":
        """
        Set the model run command.
        
        Args:
            command: Command to run the forward model
            
        Returns:
            Self for method chaining
        """
        if self._pst is None:
            raise ValueError("No Pst built yet. Call build() first.")
        
        self._pst.model_command = [command]
        return self
    
    def add_template_file(
        self, 
        template_path: Union[str, Path],
        model_input_path: Union[str, Path]
    ) -> "PstBuilder":
        """
        Add a template file.
        
        Args:
            template_path: Path to template file (.tpl)
            model_input_path: Path to model input file that template creates
            
        Returns:
            Self for method chaining
        """
        if self._pst is None:
            raise ValueError("No Pst built yet. Call build() first.")
        
        # Use model_input_data DataFrame
        new_row = pd.DataFrame({
            "pest_file": [str(template_path)],
            "model_file": [str(model_input_path)]
        })
        self._pst.model_input_data = pd.concat(
            [self._pst.model_input_data, new_row], 
            ignore_index=True
        )
        return self
    
    def add_instruction_file(
        self,
        instruction_path: Union[str, Path],
        model_output_path: Union[str, Path]
    ) -> "PstBuilder":
        """
        Add an instruction file.
        
        Args:
            instruction_path: Path to instruction file (.ins)
            model_output_path: Path to model output file to read
            
        Returns:
            Self for method chaining
        """
        if self._pst is None:
            raise ValueError("No Pst built yet. Call build() first.")
        
        # Use model_output_data DataFrame
        new_row = pd.DataFrame({
            "pest_file": [str(instruction_path)],
            "model_file": [str(model_output_path)]
        })
        self._pst.model_output_data = pd.concat(
            [self._pst.model_output_data, new_row],
            ignore_index=True
        )
        return self
