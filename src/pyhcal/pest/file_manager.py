"""
File management for PEST++ workflows.

This module handles:
- Creating PEST instruction files (.ins)
- Writing simulated output in PEST-readable format
- Generating forward run scripts
- Saving/loading observed data

PEST++ IO File Workflow
=======================

PEST++ uses template files (.tpl) and instruction files (.ins) to communicate
with the model:

1. Template File (.tpl) - Created by UCI.add_parameter_template() + write_tpl()
   - Contains parameter markers like ~lzsn~ that PEST replaces with values
   - UCI class already handles this for HSPF

2. Instruction File (.ins) - Created by create_instruction_file()
   - Tells PEST how to extract simulated values from model output
   - Uses markers like !obs_name! to identify values to read

3. Simulated Output File - Created by write_simulated_output() in forward run
   - Simple text file format that instruction file can parse
   - Written by post-processor after each model run

4. pyemu.Pst.from_io_files() - Builds PST from template + instruction files
   - Parses .tpl files to get parameter names
   - Parses .ins files to get observation names
   - Reads output files to get initial simulated values

Example Workflow:
    # 1. Create instruction file (once during setup)
    obs_data = {"flow_20050101": 100.5, "flow_20050102": 95.2, ...}
    create_instruction_file(obs_data, "model.ins")
    
    # 2. In forward_run script, after model runs:
    simulated = {"flow_20050101": 98.3, "flow_20050102": 93.1, ...}
    write_simulated_output(simulated, "model_output.out")
    
    # 3. Build PST using pyemu
    pst = pyemu.Pst.from_io_files(
        tpl_files=["model.tpl"],
        in_files=["model.uci"],
        ins_files=["model.ins"],
        out_files=["model_output.out"]
    )
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd


def create_instruction_file(
    observation_data: Dict[str, float],
    filepath: Union[str, Path],
) -> Path:
    """
    Create a PEST instruction file (.ins) for extracting observations.
    
    The instruction file tells PEST how to read simulated values from the
    model output file. Each observation gets a line that:
    1. Searches for the observation name as a marker
    2. Reads the numeric value after the marker
    
    Args:
        observation_data: Dict of observation names to target values
            The values are used to establish the observations, but only
            the names are written to the instruction file.
        filepath: Path to write instruction file
        
    Returns:
        Path to created instruction file
        
    Example:
        >>> obs = {"flow_20050101": 100.5, "nse": 0.85, "bias": -2.3}
        >>> create_instruction_file(obs, "output.ins")
        
        Creates instruction file:
            pif @
            l1 @flow_20050101@ !flow_20050101!
            l1 @nse@ !nse!
            l1 @bias@ !bias!
    """
    filepath = Path(filepath)
    
    lines = ["pif @"]  # PEST instruction file header (@ is marker character)
    for obs_name in observation_data.keys():
        # l1 = read 1 line, @name@ = search for marker, !name! = read value
        lines.append(f"l1 @{obs_name}@ !{obs_name}!")
    
    filepath.write_text("\n".join(lines) + "\n")
    return filepath


def write_simulated_output(
    simulated_values: Dict[str, float],
    filepath: Union[str, Path],
) -> Path:
    """
    Write simulated values in PEST-readable format.
    
    This should be called by the forward run script after each model run.
    The format matches what the instruction file expects.
    
    Args:
        simulated_values: Dict of observation names to simulated values
        filepath: Path to write output file
        
    Returns:
        Path to created output file
        
    Example:
        >>> sim = {"flow_20050101": 98.3, "nse": 0.82, "bias": -5.1}
        >>> write_simulated_output(sim, "model_output.out")
        
        Creates output file:
            flow_20050101  98.3
            nse  0.82
            bias  -5.1
    """
    filepath = Path(filepath)
    
    lines = []
    for obs_name, value in simulated_values.items():
        # Format: observation_name  value (space separated)
        lines.append(f"{obs_name}  {value:20.10E}")
    
    filepath.write_text("\n".join(lines) + "\n")
    return filepath


class FileManager:
    """Manages PEST-related files."""
    
    def __init__(self, pest_dir: Union[str, Path]):
        """
        Initialize file manager.
        
        Args:
            pest_dir: Directory for PEST files
        """
        self._pest_dir = Path(pest_dir)
        self._pest_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def pest_dir(self) -> Path:
        """Get the PEST directory."""
        return self._pest_dir
    
    def write_instruction_file(
        self,
        observation_names: List[str],
        filename: str = "model_output.ins"
    ) -> Path:
        """
        Create a PEST instruction file.
        
        Args:
            observation_names: List of observation names to extract
            filename: Name of instruction file
            
        Returns:
            Path to created file
        """
        filepath = self._pest_dir / filename
        
        lines = ["pif @"]  # PEST instruction file header
        for obs_name in observation_names:
            # Read each line, extract the value after the observation name
            lines.append(f"l1 @{obs_name}@ !{obs_name}!")
        
        filepath.write_text("\n".join(lines))
        return filepath
    
    def write_output_file_template(
        self,
        observation_names: List[str],
        filename: str = "model_output.out"
    ) -> Path:
        """
        Create a template for the model output file format.
        
        The forward run script should write output in this format.
        
        Args:
            observation_names: List of observation names
            filename: Name of output file
            
        Returns:
            Path to created file
        """
        filepath = self._pest_dir / filename
        
        lines = []
        for obs_name in observation_names:
            lines.append(f"{obs_name} 0.0")  # Placeholder values
        
        filepath.write_text("\n".join(lines))
        return filepath
    
    def write_forward_run_script(
        self,
        model_command: str,
        observation_names: List[str],
        observed_data_file: str = "observed_data.csv",
        output_file: str = "model_output.out",
        filename: str = "forward_run.py"
    ) -> Path:
        """
        Generate the forward run script.
        
        Args:
            model_command: Command to run HSPF model
            observation_names: List of observation names
            observed_data_file: File with observed data
            output_file: File to write simulated values
            filename: Name of script file
            
        Returns:
            Path to created file
        """
        filepath = self._pest_dir / filename
        
        script = f'''#!/usr/bin/env python
"""
PEST++ forward model run script.

This script is called by PEST++ for each model evaluation:
1. Runs the HSPF model
2. Extracts simulated values
3. Calculates statistics
4. Writes output for PEST to read
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Add pyhcal to path if needed
try:
    from pyhcal.forward_run import run_forward_model, ForwardModelConfig
    from pyhcal import metrics
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pyhcal.forward_run import run_forward_model, ForwardModelConfig
    from pyhcal import metrics


def main():
    """Run the forward model and write output."""
    config = ForwardModelConfig(
        model_command="{model_command}",
        observed_data_file="{observed_data_file}",
        output_file="{output_file}",
        observation_names={observation_names!r}
    )
    
    run_forward_model(config)


if __name__ == "__main__":
    main()
'''
        filepath.write_text(script)
        filepath.chmod(0o755)  # Make executable
        return filepath
    
    def save_observed_data(
        self,
        data: Dict[str, pd.Series],
        filename: str = "observed_data.csv"
    ) -> Path:
        """
        Save observed data to CSV.
        
        Args:
            data: Dictionary of observation name -> observed series
            filename: Name of output file
            
        Returns:
            Path to created file
        """
        filepath = self._pest_dir / filename
        
        # Combine all series into a DataFrame
        df = pd.DataFrame(data)
        df.index.name = "datetime"
        df.to_csv(filepath)
        
        return filepath
    
    def load_observed_data(
        self,
        filename: str = "observed_data.csv"
    ) -> pd.DataFrame:
        """
        Load observed data from CSV.
        
        Args:
            filename: Name of input file
            
        Returns:
            DataFrame with observed data
        """
        filepath = self._pest_dir / filename
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
