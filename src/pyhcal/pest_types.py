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
    group : str
        Parameter group for PEST (e.g., 'hydrology', 'sediment')
    metadata : dict
        Additional metadata (block, table, table_id, column, etc.)
    """
    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    transform: str = 'none'
    group: str = 'pargp'
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
