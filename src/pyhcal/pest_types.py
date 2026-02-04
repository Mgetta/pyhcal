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
    group : str
        Parameter group for PEST (e.g., 'hydrology', 'sediment')
    initial : float
        Initial/starting value
    lower : float
        Lower bound
    upper : float
        Upper bound
    transform : str
        Parameter transformation ('log', 'none', 'fixed')
    block : str
        UCI file block where parameter appears
    table : str
        UCI file table within the block
    table_id : str
        Row identifier within the table
    column : str
        Column name in the table
    """
    name: str
    group: str
    initial: float
    lower: float
    upper: float
    transform: str = 'log'
    block: str = ''
    table: str = ''
    table_id: str = ''
    column: str = ''
    
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
        Short name for this observation (max 12 chars, used as prefix)
    group : str
        Observation group for PEST (e.g., 'flow', 'sediment')
    weight : float
        Base weight for observations from this location
    station_id : str
        Original station identifier (for documentation)
    constituent : str
        What's being observed (e.g., 'flow', 'tss')
    obs_type : str
        Type of observation (e.g., 'streamflow', 'concentration')
    """
    name: str
    group: str = ''
    weight: float = 1.0
    station_id: str = ''
    constituent: str = 'flow'
    obs_type: str = 'streamflow'
    
    def __post_init__(self):
        """Set defaults and validate."""
        if not self.group:
            self.group = self.name
        if len(self.name) > 12:
            raise ValueError(f"Observation name '{self.name}' exceeds 12 char PEST limit")


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
    target_values : dict
        Target values for statistics (e.g., NSE=0.7)
    """
    include_timeseries: bool = True
    include_statistics: bool = True
    include_exceedance: bool = False
    statistics: List[str] = field(default_factory=lambda: ['nse', 'bias', 'monthly_avg'])
    exceedance_probs: List[float] = field(default_factory=lambda: [10, 25, 50, 75, 90])
    target_values: Optional[dict] = None


@dataclass  
class PestConfig:
    """
    Configuration for PEST++ run.
    
    Attributes
    ----------
    case_name : str
        Name for PEST case files
    pest_mode : str
        PEST++ mode ('estimation', 'sensitivity', 'identifiability')
    svd_pack : str
        SVD package ('propack' or 'eigen')
    max_iterations : int
        Maximum calibration iterations
    """
    case_name: str = 'hspf_cal'
    pest_mode: str = 'estimation'
    svd_pack: str = 'propack'
    max_iterations: int = 20


# Type alias for observation data structure
# Format: {'location_name': (observed_series, simulated_series)}
ObservationData = dict  # More specific: Dict[str, Tuple[pd.Series, pd.Series]]
