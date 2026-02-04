"""
PEST++ Package for HSPF Model Calibration.

This package provides a simple, readable API for setting up PEST++ projects
that integrate with the pyhcal Calibrator class.

Usage:
    # PestProject is accessed through the Calibrator class
    cal.pest.create_project()
    cal.pest.add_observation('st1_flow', 'S001', 100, 'flow', weight=1.0)
    cal.pest.add_parameter('PERLND', 'PWAT-PARM2', 0, 'LZSN')
    
    obs_df = cal.pest.build_observations()
    par_df = cal.pest.build_parameters()
    
    # Use with pyemu
    pst = pyemu.Pst.from_io_files(tpl_files, in_files, ins_files, out_files)
"""

from .pest_project import PestProject, HSPF_PARAMETER_DEFAULTS

__all__ = [
    'PestProject',
    'HSPF_PARAMETER_DEFAULTS',
]
