# -*- coding: utf-8 -*-
"""
pyhcal - Python HSPF Calibration Package

Created on Thu Dec  9 10:39:01 2021

@author: mfratki
"""

# Core modules (always available)
from pyhcal.metrics import (
    calculate_all_statistics,
    nse_metric,
    log_nse_metric,
    kge_metric,
    bias_metric,
    pbias_metric,
)

# PEST subpackage - import from pest/
from pyhcal.pest import (
    # Data types (always available - no external dependencies)
    Parameter,
    ObservationConfig,
    BuildSettings,
    PestConfig,
    # Managers (always available - no external dependencies)
    ParameterManager,
    ObservationManager,
    get_hspf_hydrology_defaults,
    # Observation building
    align_timeseries,
    build_observations,
    ObservationBuilder,
)

# PEST workflow (optional - requires pyemu)
try:
    from pyhcal.pest import (
        PestWorkflow,
        PstBuilder,
        FileManager,
        create_uci_template,
        ForwardModelConfig,
        run_forward_model,
    )
except ImportError:
    # pyemu not installed
    pass
