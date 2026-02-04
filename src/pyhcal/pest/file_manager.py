"""
File management for PEST++ workflows.

This module handles:
- Creating PEST template files (.tpl)
- Creating PEST instruction files (.ins)
- Generating forward run scripts
- Saving/loading observed data
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd


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
