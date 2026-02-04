"""
Observation management for PEST++ workflows.

This module handles:
- Observation configuration storage and validation
- Observed/simulated data management
- Building PEST observations from data
"""
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from .pest_types import ObservationConfig, BuildSettings
from . import observation_builder


class ObservationManager:
    """Manages observations for PEST++ optimization."""
    
    def __init__(self, settings: Optional[BuildSettings] = None):
        """
        Initialize observation manager.
        
        Args:
            settings: Default settings for building observations
        """
        self._observations: Dict[str, ObservationConfig] = {}
        self._data: Dict[str, Dict[str, pd.Series]] = {}  # name -> {observed, simulated}
        self._settings = settings or BuildSettings()
        self._built_observations: Optional[pd.DataFrame] = None
    
    def add(
        self,
        name: str,
        group: str,
        obs_type: str = "flow",
        weight: float = 1.0,
        observed: Optional[pd.Series] = None,
        simulated: Optional[pd.Series] = None,
        **metadata
    ) -> "ObservationManager":
        """
        Add an observation location.
        
        Args:
            name: Observation name (max 12 chars recommended)
            group: Observation group for PEST
            obs_type: Type of observation (flow, sediment, etc.)
            weight: Weight for this observation in objective function
            observed: Optional observed timeseries data
            simulated: Optional simulated timeseries data
            **metadata: Additional metadata (station_id, reach_id, etc.)
            
        Returns:
            Self for method chaining
        """
        self._validate_observation_name(name)
        
        self._observations[name] = ObservationConfig(
            name=name,
            group=group,
            obs_type=obs_type,
            weight=weight,
            metadata=metadata
        )
        
        if observed is not None or simulated is not None:
            self.set_data(name, observed, simulated)
        
        return self
    
    def set_data(
        self,
        name: str,
        observed: Optional[pd.Series] = None,
        simulated: Optional[pd.Series] = None
    ) -> "ObservationManager":
        """
        Set observed/simulated data for an observation.
        
        Args:
            name: Observation name
            observed: Observed timeseries
            simulated: Simulated timeseries
            
        Returns:
            Self for method chaining
        """
        if name not in self._observations:
            raise KeyError(f"Observation '{name}' not found. Add it first with add()")
        
        if name not in self._data:
            self._data[name] = {}
        
        if observed is not None:
            self._data[name]["observed"] = observed
        if simulated is not None:
            self._data[name]["simulated"] = simulated
        
        # Invalidate built observations when data changes
        self._built_observations = None
        
        return self
    
    def get(self, name: str) -> ObservationConfig:
        """Get observation config by name."""
        if name not in self._observations:
            raise KeyError(f"Observation '{name}' not found")
        return self._observations[name]
    
    def get_data(self, name: str) -> Dict[str, pd.Series]:
        """Get data for an observation."""
        return self._data.get(name, {})
    
    def has_data(self, name: str) -> bool:
        """Check if observation has both observed and simulated data."""
        data = self._data.get(name, {})
        return "observed" in data and "simulated" in data
    
    def remove(self, name: str) -> "ObservationManager":
        """Remove an observation."""
        if name in self._observations:
            del self._observations[name]
        if name in self._data:
            del self._data[name]
        return self
    
    def __len__(self) -> int:
        return len(self._observations)
    
    def __iter__(self) -> Iterator[ObservationConfig]:
        return iter(self._observations.values())
    
    def __contains__(self, name: str) -> bool:
        return name in self._observations
    
    @property
    def names(self) -> List[str]:
        """List of observation names."""
        return list(self._observations.keys())
    
    def build_observations(
        self,
        settings: Optional[BuildSettings] = None
    ) -> pd.DataFrame:
        """
        Build PEST observations from all observations with data.
        
        Args:
            settings: Override default build settings
            
        Returns:
            DataFrame of PEST observations
        """
        settings = settings or self._settings
        all_obs = []
        
        for name, config in self._observations.items():
            if not self.has_data(name):
                continue
            
            data = self._data[name]
            obs_df = self._build_single_observation(
                name=name,
                config=config,
                observed=data["observed"],
                simulated=data["simulated"],
                settings=settings
            )
            all_obs.append(obs_df)
        
        if not all_obs:
            return pd.DataFrame()
        
        self._built_observations = pd.concat(all_obs, ignore_index=True)
        return self._built_observations
    
    def get_built_observations(self) -> Optional[pd.DataFrame]:
        """Get previously built observations (or None if not built)."""
        return self._built_observations
    
    def _build_single_observation(
        self,
        name: str,
        config: ObservationConfig,
        observed: pd.Series,
        simulated: pd.Series,
        settings: BuildSettings
    ) -> pd.DataFrame:
        """Build PEST observations for a single location."""
        return observation_builder.build_observations(
            observed=observed,
            simulated=simulated,
            prefix=name,
            include_timeseries=settings.include_timeseries,
            include_statistics=settings.include_statistics,
            include_exceedance=settings.include_exceedance,
            statistics=settings.statistics,
            exceedance_probs=settings.exceedance_probs,
            timeseries_weight=config.weight,
            statistics_weight=config.weight * 10,  # Upweight stats
            exceedance_weight=config.weight * 5,
            target_values=settings.stat_targets,
            verbose=False
        )
    
    @staticmethod
    def _validate_observation_name(name: str) -> None:
        """Validate observation name."""
        if not name:
            raise ValueError("Observation name cannot be empty")
        # PEST names have limits, but we allow longer for prefixes
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Observation name '{name}' contains invalid characters")
