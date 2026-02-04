"""
PEST++ Workflow Package for HSPF Model Calibration.

This package provides tools for setting up PEST++ parameter estimation
workflows with HSPF models using pyemu.

Architecture:
    pest_types.py         - Data classes (Parameter, ObservationConfig, etc.)
    parameter_manager.py  - Parameter storage and HSPF defaults
    observation_manager.py - Observation storage and data management
    observation_builder.py - Build PEST observations from timeseries
    file_manager.py       - Template/instruction file handling
    forward_run.py        - Model execution for PEST iterations
    pest_workflow.py      - Main coordinator class (uses pyemu.Pst directly)

Usage:
    from pyhcal.pest import PestWorkflow
    
    pest = PestWorkflow("my_project")
    pest.load_model(0)
    pest.add_hspf_parameter('PERLND', 'PWAT-PARM2', 0, 'LZSN')
    pest.add_observation("outlet", "flow", observed=obs, simulated=sim)
    pest.build()
    pest.write("my_case.pst")
"""

from .pest_types import (
    Parameter,
    ObservationConfig,
    BuildSettings,
    PestConfig,
)
from .parameter_manager import (
    ParameterManager,
    get_hspf_hydrology_defaults,
)
from .observation_manager import ObservationManager
from .observation_builder import (
    align_timeseries,
    build_timeseries_observations,
    build_statistic_observations,
    build_exceedance_observations,
    build_observations,
    ObservationBuilder,
)
from .file_manager import (
    FileManager,
    create_instruction_file,
    write_simulated_output,
    write_observation_metadata,
    read_observation_metadata,
    apply_observation_metadata,
)
from .forward_run import (
    ForwardModelConfig,
    run_hspf,
    load_observed_data,
    extract_simulated_data,
    align_timeseries as align_timeseries_forward,
    calculate_observation_values,
    write_pest_output,
    run_forward_model,
)
from .pest_workflow import PestWorkflow

__all__ = [
    # Data types
    'Parameter',
    'ObservationConfig',
    'BuildSettings',
    'PestConfig',
    # Managers
    'ParameterManager',
    'ObservationManager',
    # Builders
    'ObservationBuilder',
    'FileManager',
    # IO File functions (for pyemu.Pst.from_io_files workflow)
    'create_instruction_file',
    'write_simulated_output',
    'write_observation_metadata',
    'read_observation_metadata',
    'apply_observation_metadata',
    # Main workflow
    'PestWorkflow',
    # Utility functions
    'get_hspf_hydrology_defaults',
    'align_timeseries',
    'build_timeseries_observations',
    'build_statistic_observations',
    'build_exceedance_observations',
    'build_observations',
    # Forward model
    'ForwardModelConfig',
    'run_hspf',
    'load_observed_data',
    'extract_simulated_data',
    'calculate_observation_values',
    'write_pest_output',
    'run_forward_model',
]
