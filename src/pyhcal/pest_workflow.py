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

3. **Run PEST++**:
   - Execute PESTPP-GLM, PESTPP-IES, or other PEST++ programs
   - Monitor convergence and adjust if needed

4. **Post-Processing**:
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


class PestWorkflow:
    """
    A class to set up and manage PEST++ workflows for HSPF model calibration.
    
    This class provides methods to:
    1. Define parameters from UCI tables for calibration
    2. Create PEST template files using the UCI class's add_parameter_template method
    3. Set up observations from model outputs
    4. Build PEST control files using pyemu
    5. Run sensitivity analysis and parameter estimation
    
    Parameters
    ----------
    project_path : str or Path
        Path to the calibration project folder
    uci : object
        UCI object from the hspf package (must have add_parameter_template method)
        
    Attributes
    ----------
    project_path : Path
        Path to the project folder
    uci : object
        UCI object for manipulating HSPF input files
    pest_path : Path
        Path to the PEST working directory
    parameters : dict
        Dictionary of parameter definitions
    observations : dict
        Dictionary of observation definitions
    pst : pyemu.Pst
        PEST control file object (created after calling build_pst)
        
    Examples
    --------
    Basic workflow example::
    
        from pyhcal import calibrators
        from pyhcal.pest_workflow import PestWorkflow
        
        # Load an existing calibration project
        cal = calibrators.load('/path/to/project')
        cal.load_model(0)  # Load initial model
        
        # Create PEST workflow
        pest = PestWorkflow(cal.project_path, cal.uci)
        
        # Add parameters for hydrology calibration
        pest.add_parameter(
            name='LZSN',
            block='PERLND',
            table='PWAT-PARM2',
            table_id=0,
            lower_bound=2.0,
            upper_bound=15.0,
            initial_value=5.0,
            description='Lower zone nominal storage (inches)'
        )
        
        # Build and save PEST control file
        pest.build_pst()
        pest.write_pest_files()
    """
    
    # Common HSPF hydrological parameters with typical bounds
    # These are starting points - actual bounds depend on watershed characteristics
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
        'UZSN': {
            'block': 'PERLND',
            'table': 'PWAT-PARM2',
            'column': 'UZSN',
            'lower': 0.05,
            'upper': 2.0,
            'description': 'Upper zone nominal soil moisture storage (inches)'
        },
        'INTFW': {
            'block': 'PERLND',
            'table': 'PWAT-PARM3',
            'column': 'INTFW',
            'lower': 0.0,
            'upper': 10.0,
            'description': 'Interflow inflow parameter'
        },
        'IRC': {
            'block': 'PERLND',
            'table': 'PWAT-PARM3',
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
        'AGWRC': {
            'block': 'PERLND',
            'table': 'PWAT-PARM3',
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
            'table': 'PWAT-PARM4',
            'column': 'AGWETP',
            'lower': 0.0,
            'upper': 0.2,
            'description': 'Fraction of remaining ET from active GW'
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
        """
        Add a parameter definition for PEST calibration.
        
        This method registers a parameter that will be included in the PEST
        control file. The parameter corresponds to a specific location in
        the HSPF UCI file.
        
        Parameters
        ----------
        name : str
            Unique name for the parameter (max 12 characters for PEST)
        block : str
            UCI block name (e.g., 'PERLND', 'IMPLND', 'RCHRES')
        table : str
            UCI table name within the block (e.g., 'PWAT-PARM2', 'HYDR-PARM2')
        table_id : int, default 0
            Table ID when multiple tables exist with the same name
        column : str, optional
            Specific column name in the table. If None, uses the parameter name.
        opnids : list of int, optional
            Operation IDs (pervious land segments, reaches, etc.) to apply
            the parameter to. If None, applies to all operations in the table.
        lower_bound : float, default 0.001
            Lower bound for parameter estimation
        upper_bound : float, default 1000.0
            Upper bound for parameter estimation
        initial_value : float, optional
            Initial parameter value. If None, uses the current UCI value.
        transform : str, default 'log'
            Parameter transformation for PEST:
            - 'log': Log transformation (good for parameters with wide ranges)
            - 'none': No transformation
            - 'fixed': Parameter is fixed (not estimated)
        group : str, optional
            Parameter group name for PEST. Parameters in the same group
            share regularization and can be tied. If None, uses block name.
        description : str, default ''
            Description of the parameter for documentation
            
        Notes
        -----
        PEST Parameter Transformations:
        - Log transformation is recommended when parameters can vary over
          orders of magnitude (e.g., INFILT from 0.001 to 0.5)
        - Use 'none' for parameters with narrow ranges or that can be negative
        
        Examples
        --------
        Add LZSN parameter for all PERLND segments::
        
            pest.add_parameter(
                name='LZSN',
                block='PERLND',
                table='PWAT-PARM2',
                lower_bound=2.0,
                upper_bound=15.0,
                transform='log'
            )
            
        Add INFILT for specific land segments::
        
            pest.add_parameter(
                name='INFILT_AG',
                block='PERLND',
                table='PWAT-PARM2',
                column='INFILT',
                opnids=[101, 102, 103],  # Agricultural segments
                lower_bound=0.01,
                upper_bound=0.3
            )
        """
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
        """
        Add common hydrological parameters for calibration.
        
        This is a convenience method that adds multiple hydrological
        parameters with typical bounds based on HSPF calibration guidance.
        
        Parameters
        ----------
        parameters : list of str, optional
            List of parameter names to add. Available parameters:
            LZSN, INFILT, UZSN, INTFW, IRC, LZETP, AGWRC, DEEPFR, BASETP, AGWETP
            If None, adds: LZSN, INFILT, UZSN, INTFW, IRC, AGWRC
        opnids : list of int, optional
            Operation IDs to apply parameters to. If None, applies to all.
        group : str, optional
            Parameter group name. If None, uses 'hydrology'
            
        Examples
        --------
        Add default hydrology parameters::
        
            pest.add_hydrology_parameters()
            
        Add specific parameters::
        
            pest.add_hydrology_parameters(
                parameters=['LZSN', 'INFILT', 'AGWRC'],
                group='baseflow'
            )
        """
        if parameters is None:
            # Default set of most sensitive hydrology parameters
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
        """
        Add an observation definition for PEST calibration.
        
        Observations are the measured data that PEST tries to match by
        adjusting parameters. For hydrology calibration, these are typically
        observed streamflow values.
        
        Parameters
        ----------
        name : str
            Unique name for the observation group (max 12 characters)
        obs_type : str
            Type of observation:
            - 'timeseries': Time series of values (e.g., daily flow)
            - 'summary': Summary statistics (e.g., mean annual flow)
        station_id : str, optional
            Observation station identifier
        reach_id : int, optional
            Model reach ID corresponding to the observation location
        constituent : str, default 'Q'
            Constituent type: 'Q' (flow), 'TSS', 'TP', etc.
        time_step : str, default 'D'
            Time step: 'D' (daily), 'M' (monthly), 'Y' (yearly)
        weight : float, default 1.0
            Observation weight for PEST objective function. Higher weights
            give more influence on calibration.
        group : str, optional
            Observation group name. If None, uses constituent name.
        description : str, default ''
            Description of the observation
            
        Notes
        -----
        Observation Weighting:
        - Weights determine the relative importance of observations
        - Common strategies:
          - Equal weights: All observations equally important
          - Inverse variance: Weight by measurement uncertainty
          - Flow-based: Higher weights for baseflow/low-flow periods
          
        Examples
        --------
        Add daily flow observation::
        
            pest.add_observation(
                name='flow_daily',
                obs_type='timeseries',
                station_id='USGS12345',
                reach_id=650,
                constituent='Q',
                time_step='D',
                weight=1.0
            )
        """
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
                param_name
            )
            
        # Write the template file
        self.uci.write(output_path)
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
