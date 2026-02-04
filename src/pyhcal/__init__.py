# -*- coding: utf-8 -*-
"""
pyhcal - Python HSPF Calibration Package

Created on Thu Dec  9 10:39:01 2021

@author: mfratki
"""

# Core modules
from pyhcal.metrics import (
    calculate_all_statistics,
    nse_metric,
    log_nse_metric,
    kge_metric,
    bias_metric,
    pbias_metric,
)

from pyhcal.observation_builder import (
    align_timeseries,
    build_observations,
    ObservationBuilder,
)

# PEST workflow (optional - requires pyemu)
try:
    from pyhcal.pest_workflow import PestWorkflow
    from pyhcal.forward_run import ForwardModelConfig, run_forward_model
except ImportError:
    # pyemu not installed
    pass
