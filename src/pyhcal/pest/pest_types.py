# -*- coding: utf-8 -*-
"""
PEST Data Types Module

This module defines simple data classes for PEST++ workflow.
These are plain data containers with minimal behavior - they exist
to make the code more type-safe and self-documenting.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class Parameter:
    """
    A parameter to be estimated by PEST++.
    
    Attributes
    ----------
    name : str
        Parameter name (max 12 chars for PEST compatibility)
    initial_value : float
        Initial/starting value
    lower_bound : float
        Lower bound
    upper_bound : float
        Upper bound
    transform : str
        Parameter transformation ('log', 'none', 'fixed')
    pargp : str
        Parameter group for PEST (e.g., 'hydrology', 'sediment')
    block : str
        UCI block name (e.g., 'PERLND', 'IMPLND')
    table : str
        UCI table name (e.g., 'PWAT-PARM2')
    table_id : int
        UCI table ID (usually 0 for hydrology tables)
    opnids : Optional[List[int]]
        Which operation IDs to apply parameter to. None = all operations.
        Example: [100, 200] applies only to PERLND 100 and 200
    single_template : bool
        If True, one parameter applies to all operations (e.g., 'lzsn')
        If False, each operation gets unique param (e.g., 'lzsn100', 'lzsn200')
    template_group : Optional[str]
        Character to append for grouping operations that share a parameter.
        Used with opnids to create groups: lzsn + group='a' -> 'lzsna'
    metadata : dict
        Additional metadata (column, etc.)
    
    Examples
    --------
    # Single parameter for ALL perlnds:
    Parameter('lzsn', 5.0, 2.0, 15.0, block='PERLND', table='PWAT-PARM2',
              single_template=True)
    
    # Unique parameter per perlnd (lzsn100, lzsn200, ...):
    Parameter('lzsn', 5.0, 2.0, 15.0, block='PERLND', table='PWAT-PARM2',
              single_template=False)
    
    # Grouped: upland perlnds share 'lzsna', lowland share 'lzsnb':
    Parameter('lzsn', 5.0, 2.0, 15.0, block='PERLND', table='PWAT-PARM2',
              opnids=[100,101,102], single_template=True, template_group='a')
    Parameter('lzsn', 6.0, 2.0, 15.0, block='PERLND', table='PWAT-PARM2',
              opnids=[200,201,202], single_template=True, template_group='b')
    """
    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    transform: str = 'none'
    pargp: str = 'pargp'
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate parameter name length."""
        if len(self.name) > 12:
            raise ValueError(f"Parameter name '{self.name}' exceeds 12 char PEST limit")


@dataclass
class ObservationConfig:
    """
    Configuration for a monitoring location/observation point.
    
    This stores metadata about where observations come from,
    but NOT the actual timeseries data. Data is passed separately
    to allow for flexible workflows.
    
    Attributes
    ----------
    name : str
        Short name for this observation (used as prefix in PEST)
    group : str
        Observation group for PEST (e.g., 'flow', 'sediment')
    obs_type : str
        Type of observation (e.g., 'flow', 'tss', 'sediment')
    weight : float
        Base weight for observations from this location
    metadata : dict
        Additional metadata (station_id, reach_id, constituent, etc.)
    """
    name: str
    group: str = ''
    obs_type: str = 'flow'
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Set defaults."""
        if not self.group:
            self.group = self.name


@dataclass
class BuildSettings:
    """
    Settings for building PEST observations from timeseries data.
    
    Attributes
    ----------
    include_timeseries : bool
        Include 1-to-1 timeseries observations
    include_statistics : bool
        Include summary statistic observations (NSE, bias, etc.)
    include_exceedance : bool
        Include flow duration curve observations
    statistics : List[str]
        Which statistics to include
    exceedance_probs : List[float]
        Which exceedance probabilities to include
    stat_targets : dict
        Target values for statistics (e.g., {'nse': 0.7})
    """
    include_timeseries: bool = True
    include_statistics: bool = True
    include_exceedance: bool = False
    statistics: List[str] = field(default_factory=lambda: ['nse', 'bias'])
    exceedance_probs: List[float] = field(default_factory=lambda: [10, 25, 50, 75, 90])
    stat_targets: Optional[dict] = None


@dataclass  
class PestConfig:
    """
    Configuration for PEST++ run.
    
    Attributes
    ----------
    noptmax : int
        Maximum iterations for optimization
    svd_maxsing : int
        Maximum singular values
    svd_eigthresh : float
        Eigenvalue threshold
    pestpp_options : dict
        Additional PEST++ options
    """
    noptmax: int = 20
    svd_maxsing: int = 100
    svd_eigthresh: float = 1e-6
    pestpp_options: dict = field(default_factory=dict)


# Type alias for observation data structure
# Format: {'location_name': (observed_series, simulated_series)}
ObservationData = dict  # More specific: Dict[str, Tuple[pd.Series, pd.Series]]
