#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEST++ Workflow Example for HSPF Hydrology Calibration
=======================================================

This example demonstrates how to set up a PEST++ parameter estimation
workflow for calibrating the hydrology portion of an HSPF model using
the pyhcal package and pyemu.

This is a step-by-step tutorial designed to introduce the concepts
and show the typical workflow.

Prerequisites
-------------
1. An existing pyhcal calibration project
2. The hspf package installed
3. pyemu installed (pip install pyemu)
4. PEST++ binaries installed (available from https://github.com/usgs/pestpp)

What is PEST++?
---------------
PEST++ is a software suite for parameter estimation, uncertainty quantification,
and sensitivity analysis. It's widely used in environmental modeling to:

1. **Calibrate models**: Automatically adjust parameters to match observed data
2. **Sensitivity analysis**: Identify which parameters most influence model output
3. **Uncertainty quantification**: Understand prediction uncertainty

Key PEST++ programs:
- PESTPP-GLM: Gradient-based optimization (like classic PEST)
- PESTPP-IES: Iterative Ensemble Smoother for data assimilation
- PESTPP-SEN: Global sensitivity analysis

What is pyemu?
--------------
pyemu is a Python package that provides tools for working with PEST/PEST++:
- Creating PEST input files programmatically
- Analyzing PEST output files
- Utilities for model-independent calibration

Step-by-Step Workflow
--------------------
1. Load your calibration project
2. Define parameters to estimate
3. Define observations (calibration targets)
4. Create PEST template files
5. Build the PEST control file
6. Run PEST++ (externally or via subprocess)
7. Analyze results

Author: pyhcal contributors
"""

# =============================================================================
# STEP 0: Imports
# =============================================================================
print("=" * 60)
print("PEST++ Workflow for HSPF Hydrology Calibration")
print("=" * 60)
print()

from pathlib import Path
import pandas as pd
import numpy as np

# pyhcal imports
# from pyhcal import calibrators
# from pyhcal.pest_workflow import PestWorkflow, create_pest_workflow

# For this example, we'll demonstrate the concepts with mock objects
# since we don't have an actual HSPF project to run

print("Step 0: Imports complete")
print()

# =============================================================================
# STEP 1: Understanding the UCI Class and add_parameter_template
# =============================================================================
print("=" * 60)
print("STEP 1: Understanding the UCI Class")
print("=" * 60)
print("""
The UCI (User Control Input) class in the hspf package represents the
HSPF model input file. It contains methods to:

1. Read UCI file structure (blocks, tables, parameters)
2. Modify parameter values
3. Write modified UCI files
4. Create PEST template files

Key method for PEST++:
    uci.add_parameter_template(BLOCK, TABLE, TABLEID, PARAMETER)
    
This method marks a parameter location in the UCI file for PEST to modify.

Example:
    # Mark LZSN in PERLND PWAT-PARM2 table for calibration
    uci.add_parameter_template('PERLND', 'PWAT-PARM2', 0, 'LZSN')
    
The template file will have PEST markers like:
    ~ LZSN      ~
    
where the spaces maintain the proper field width for the UCI format.
""")

# =============================================================================
# STEP 2: Define Parameters for Calibration
# =============================================================================
print("=" * 60)
print("STEP 2: Selecting Parameters for Hydrology Calibration")
print("=" * 60)
print("""
For hydrology calibration, we typically focus on these parameters:

WATER BALANCE (runoff volume):
- LZSN (2-15 in): Lower Zone Nominal Storage
  -> Higher = more interflow, less surface runoff, lower peaks
- INFILT (0.001-0.5 in/hr): Infiltration Index
  -> Higher = more infiltration, less surface runoff
- UZSN (0.05-2.0 in): Upper Zone Nominal Storage
  -> Higher = more ET, less interflow

TIMING AND FLOW COMPONENTS:
- INTFW (0-10): Interflow Index
  -> Higher = more interflow relative to GW
- IRC (0.3-0.85): Interflow Recession Constant
  -> Higher = slower interflow recession
- AGWRC (0.85-0.999): Groundwater Recession Constant
  -> Higher = slower baseflow recession

EVAPOTRANSPIRATION:
- LZETP (0-0.9): Lower Zone ET Parameter
- BASETP (0-0.2): Baseflow ET Parameter
- AGWETP (0-0.2): Active GW ET Parameter

Parameter Selection Strategy:
1. Start with few parameters (3-5) most sensitive for your watershed
2. Group related parameters (all AGWRC, all LZSN for similar land covers)
3. Consider physical constraints and parameter correlations
""")

# Example parameter table
param_df = pd.DataFrame({
    'Parameter': ['LZSN', 'INFILT', 'UZSN', 'INTFW', 'IRC', 'AGWRC'],
    'Table': ['PWAT-PARM2', 'PWAT-PARM2', 'PWAT-PARM2', 'PWAT-PARM3', 'PWAT-PARM3', 'PWAT-PARM3'],
    'Lower': [2.0, 0.001, 0.05, 0.0, 0.3, 0.85],
    'Upper': [15.0, 0.5, 2.0, 10.0, 0.85, 0.999],
    'Transform': ['log', 'log', 'log', 'none', 'none', 'log'],
    'Typical_Effect': [
        'Water balance, peak flows',
        'Infiltration vs runoff',
        'Quick flow response',
        'Interflow amount',
        'Interflow timing',
        'Baseflow recession'
    ]
})
print("\nRecommended Starting Parameters:")
print(param_df.to_string(index=False))
print()

# =============================================================================
# STEP 3: Setting Up the PEST Workflow
# =============================================================================
print("=" * 60)
print("STEP 3: Creating the PEST Workflow")
print("=" * 60)
print("""
Here's the code to set up a PEST workflow with pyhcal:

```python
from pyhcal import calibrators
from pyhcal.pest_workflow import create_pest_workflow

# Load your calibration project
project_path = '/path/to/your/calibration/project'
cal = calibrators.load(project_path)
cal.load_model(0)  # Load the initial model

# Create PEST workflow object
pest = create_pest_workflow(project_path, uci=cal.uci)

# Method 1: Add individual parameters with custom settings
pest.add_parameter(
    name='LZSN',           # Parameter name (max 12 chars)
    block='PERLND',        # UCI block
    table='PWAT-PARM2',    # UCI table
    table_id=0,            # Table instance (usually 0)
    lower_bound=2.0,       # Lower bound for estimation
    upper_bound=15.0,      # Upper bound for estimation
    initial_value=5.0,     # Starting value (or use UCI value)
    transform='log',       # 'log' or 'none'
    group='hydrology',     # Parameter group for PEST
    description='Lower zone nominal storage (inches)'
)

# Method 2: Use convenience method for common hydrology parameters
pest.add_hydrology_parameters(
    parameters=['LZSN', 'INFILT', 'AGWRC'],  # Select specific params
    opnids=None,          # Apply to all land segments (or list specific IDs)
    group='hydrology'
)

# View parameter summary
print(pest.get_parameter_summary())
```
""")

# =============================================================================
# STEP 4: Adding Observations
# =============================================================================
print("=" * 60)
print("STEP 4: Defining Observations (Calibration Targets)")
print("=" * 60)
print("""
Observations are the measured data that PEST tries to match.
For hydrology, this is typically streamflow.

```python
# Add daily flow observation at a gauging station
pest.add_observation(
    name='flow_daily',     # Observation name
    obs_type='timeseries', # 'timeseries' or 'summary'
    station_id='USGS12345',# Station identifier
    reach_id=650,          # HSPF reach number
    constituent='Q',       # 'Q' for flow
    time_step='D',         # 'D' daily, 'M' monthly, 'Y' yearly
    weight=1.0,            # Observation weight
    group='flow'           # Observation group
)

# View observation summary
print(pest.get_observation_summary())
```

Observation Weighting Strategies:
---------------------------------
1. Equal weights: All observations equally important
   -> Simple, but may overweight low-flow periods with many data points

2. Inverse variance: Weight by measurement uncertainty
   -> weight = 1 / std(measurement_error)
   -> Reduces influence of uncertain measurements

3. Flow-based: Different weights for flow regimes
   -> Higher weights for baseflow (often harder to match)
   -> Lower weights for peak flows (often uncertain)

4. Custom: Based on calibration goals
   -> Weight NSE, percent bias, or specific flow statistics
""")

# =============================================================================
# STEP 5: Creating PEST Files
# =============================================================================
print("=" * 60)
print("STEP 5: Building and Writing PEST Files")
print("=" * 60)
print("""
```python
# Build the PEST control file
pst = pest.build_pst(
    name='hydro_cal',      # Case name
    algorithm='glm',       # 'glm', 'ies', or 'sen'
    noptmax=30,            # Max iterations
    phiredswh=0.1          # Convergence threshold
)

# Optionally modify PST object (advanced)
# pst.parameter_data.loc['LZSN', 'parlbnd'] = 3.0
# pst.observation_data.loc['flow_daily', 'weight'] = 2.0

# Write all PEST files
pst_path = pest.write_pest_files('hydro_cal')
print(f"PEST files written to: {pst_path}")
```

Files Created:
--------------
- hydro_cal.pst: PEST control file
- hydro_cal.tpl: Template file for UCI input
- hydro_cal.ins: Instruction file for model output

These files define:
1. What parameters to adjust and their bounds
2. What observations to match
3. How to run the model
4. Algorithm settings
""")

# =============================================================================
# STEP 6: Running PEST++
# =============================================================================
print("=" * 60)
print("STEP 6: Running PEST++")
print("=" * 60)
print("""
PEST++ can be run from the command line or via subprocess:

Command Line:
-------------
# Navigate to PEST directory
cd /path/to/project/pest

# Run PESTPP-GLM (gradient-based estimation)
pestpp-glm hydro_cal.pst

# Run PESTPP-IES (ensemble smoother)
pestpp-ies hydro_cal.pst

# Run PESTPP-SEN (sensitivity analysis)
pestpp-sen hydro_cal.pst

From Python (subprocess):
-------------------------
```python
import subprocess

# Run PEST++
result = subprocess.run(
    ['pestpp-glm', 'hydro_cal.pst'],
    cwd=pest.pest_path,
    capture_output=True,
    text=True
)
print(result.stdout)
```

Tips for Running:
-----------------
1. Start with sensitivity analysis (pestpp-sen) to identify
   which parameters actually matter for your watershed

2. For initial calibration, use pestpp-glm with modest noptmax (20-30)

3. For uncertainty analysis, use pestpp-ies with 50-100 realizations

4. Monitor the .rec file for convergence and any issues
""")

# =============================================================================
# STEP 7: Interpreting Results
# =============================================================================
print("=" * 60)
print("STEP 7: Analyzing PEST++ Results")
print("=" * 60)
print("""
Output Files to Check:
----------------------
1. hydro_cal.rec: Main record file with all details
2. hydro_cal.par: Estimated parameter values
3. hydro_cal.sen: Parameter sensitivities
4. hydro_cal.iobj: Objective function by iteration
5. hydro_cal.jco: Jacobian matrix (parameter-observation sensitivities)

Reading Results with pyemu:
---------------------------
```python
import pyemu

# Load results
pst = pyemu.Pst('hydro_cal.pst')

# Get optimized parameters
par_data = pst.parameter_data
print("Optimized parameters:")
print(par_data[['parval1', 'parlbnd', 'parubnd']])

# Check objective function
phi = pst.phi  # Final objective function value
print(f"Final objective function: {phi}")

# Load Jacobian for sensitivity analysis
jco = pyemu.Jco.from_binary('hydro_cal.jcb')
# or
pst_result = pyemu.Pst('hydro_cal.pst')
# Check sensitivity
```

Sensitivity Interpretation:
---------------------------
From pestpp-sen output:
- Mean: Average effect of parameter on output
- Std: Variation in effect (indicates non-linearity/interactions)

High sensitivity: Parameter strongly affects model output
  -> Good candidate for calibration
  
Low sensitivity: Parameter has little effect
  -> Consider fixing to reduce parameter space
  
High std/mean ratio: Non-linear effects or interactions
  -> May need careful bounds or regularization
""")

# =============================================================================
# STEP 8: Advanced Topics
# =============================================================================
print("=" * 60)
print("STEP 8: Advanced Topics")
print("=" * 60)
print("""
Parameterization Strategies:
----------------------------
1. Single-value parameters:
   - One LZSN value for all pervious land segments
   - Simpler, fewer parameters, faster calibration
   
2. Land-cover based:
   - Different LZSN for forest vs agriculture vs urban
   - Physically meaningful, captures land use variability
   
3. Spatial multipliers:
   - Base value × spatial multiplier for each segment
   - Preserves relative spatial patterns

4. Tied parameters:
   - LZSN_forest = 1.2 × LZSN_ag (enforce ratio)
   - Reduces effective parameters while maintaining relationships

Regularization (for overparameterized problems):
------------------------------------------------
When you have more parameters than observations can constrain:
- Tikhonov regularization: Keep parameters near initial values
- SVD-assist: Remove insensitive parameter combinations
- Preferred value constraints: Penalize deviation from prior knowledge

Pilot Points (for distributed parameters):
------------------------------------------
For spatially varying parameters that need smooth interpolation:
```python
# pyemu has utilities for pilot points
pp_df = pyemu.pp_utils.setup_pilotpoints_grid(...)
```

Multi-objective Calibration:
----------------------------
If matching multiple aspects (NSE, bias, low-flow, high-flow):
- Define observation groups with different weights
- Or use Pareto-optimal approaches

Common Issues:
--------------
1. Parameter at bounds: Bounds may be too restrictive
2. Poor fit despite many iterations: 
   - Wrong parameters selected
   - Model structure inadequate
3. Oscillating objective function: 
   - Correlated parameters
   - Try SVD-assist or regularization
""")

# =============================================================================
# Complete Example Script
# =============================================================================
print("=" * 60)
print("COMPLETE EXAMPLE SCRIPT")
print("=" * 60)
print("""
# Complete working example (save as run_pest_calibration.py)

from pathlib import Path
from pyhcal import calibrators
from pyhcal.pest_workflow import create_pest_workflow

def main():
    # Configuration
    PROJECT_PATH = '/path/to/your/calibration/project'
    MODEL_RUN = 0  # Initial model run number
    REACH_ID = 650  # Calibration reach
    STATION_ID = 'USGS12345678'  # Observation station
    
    # Load project and model
    print("Loading calibration project...")
    cal = calibrators.load(PROJECT_PATH)
    cal.load_model(MODEL_RUN)
    
    # Create PEST workflow
    print("Setting up PEST workflow...")
    pest = create_pest_workflow(PROJECT_PATH, uci=cal.uci)
    
    # Add hydrology parameters
    pest.add_hydrology_parameters(
        parameters=['LZSN', 'INFILT', 'UZSN', 'AGWRC', 'IRC', 'INTFW']
    )
    
    # Add flow observation
    pest.add_observation(
        name='flow',
        obs_type='timeseries',
        station_id=STATION_ID,
        reach_id=REACH_ID,
        constituent='Q',
        weight=1.0
    )
    
    # Build PEST control file
    print("Building PEST control file...")
    pst = pest.build_pst(
        name='hydro_cal',
        algorithm='glm',
        noptmax=30
    )
    
    # Write files
    pst_path = pest.write_pest_files('hydro_cal')
    print(f"PEST files written to: {pst_path}")
    
    # Print summary
    print("\\nParameter Summary:")
    print(pest.get_parameter_summary())
    
    print("\\nObservation Summary:")
    print(pest.get_observation_summary())
    
    print("\\n" + "="*60)
    print("Next Steps:")
    print("1. Review the .pst file and adjust as needed")
    print("2. Run sensitivity analysis: pestpp-sen hydro_cal.pst")
    print("3. Run calibration: pestpp-glm hydro_cal.pst")
    print("4. Analyze results in the .rec, .par, .sen files")
    print("="*60)

if __name__ == '__main__':
    main()
""")

print("=" * 60)
print("END OF TUTORIAL")
print("=" * 60)
print("""
Next Steps:
-----------
1. Install PEST++ binaries from: https://github.com/usgs/pestpp/releases
2. Try the PestWorkflow class with your actual calibration project
3. Start with sensitivity analysis to understand parameter importance
4. Begin with a simple calibration (few parameters) and add complexity

For questions or issues, consult:
- PEST++ documentation: https://github.com/usgs/pestpp
- pyemu documentation: https://github.com/pypest/pyemu
- HSPF User's Manual for parameter guidance
""")
