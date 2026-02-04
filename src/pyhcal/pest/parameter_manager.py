"""
Parameter management for PEST++ workflows.

This module handles:
- Parameter storage and validation
- HSPF hydrology parameter defaults
- Parameter transformation settings
"""
from typing import Dict, Iterator, List, Optional, Tuple
import pandas as pd

from .pest_types import Parameter


class ParameterManager:
    """Manages parameters for PEST++ optimization."""
    
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
    
    def add(
        self,
        name: str,
        initial_value: float,
        lower_bound: float,
        upper_bound: float,
        transform: str = "none",
        group: Optional[str] = None,
        **metadata
    ) -> "ParameterManager":
        """
        Add a parameter.
        
        Args:
            name: Parameter name (must be unique, max 12 chars for PEST)
            initial_value: Starting value
            lower_bound: Minimum allowed value
            upper_bound: Maximum allowed value
            transform: Transformation type ("none", "log", "fixed")
            group: Parameter group name
            **metadata: Additional metadata (block, table, table_id, etc.)
            
        Returns:
            Self for method chaining
        """
        self._validate_parameter_name(name)
        self._validate_bounds(name, initial_value, lower_bound, upper_bound)
        
        self._parameters[name] = Parameter(
            name=name,
            initial_value=initial_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=transform,
            group=group or "pargp",
            metadata=metadata
        )
        return self
    
    def add_from_defaults(
        self, 
        defaults: Dict[str, Dict], 
        only: Optional[List[str]] = None
    ) -> "ParameterManager":
        """
        Add parameters from a defaults dictionary.
        
        Args:
            defaults: Dictionary of parameter definitions
            only: If provided, only add these parameters
            
        Returns:
            Self for method chaining
        """
        for name, config in defaults.items():
            if only is not None and name not in only:
                continue
            self.add(name=name, **config)
        return self
    
    def get(self, name: str) -> Parameter:
        """Get a parameter by name."""
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        return self._parameters[name]
    
    def remove(self, name: str) -> "ParameterManager":
        """Remove a parameter."""
        if name in self._parameters:
            del self._parameters[name]
        return self
    
    def __len__(self) -> int:
        return len(self._parameters)
    
    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters.values())
    
    def __contains__(self, name: str) -> bool:
        return name in self._parameters
    
    @property
    def names(self) -> List[str]:
        """List of parameter names."""
        return list(self._parameters.keys())
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert parameters to DataFrame (for pyemu)."""
        if not self._parameters:
            return pd.DataFrame()
        
        records = []
        for p in self._parameters.values():
            records.append({
                "parnme": p.name,
                "parval1": p.initial_value,
                "parlbnd": p.lower_bound,
                "parubnd": p.upper_bound,
                "pargp": p.group,
                "partrans": p.transform,
                "parchglim": "relative",
                "scale": 1.0,
                "offset": 0.0,
            })
        return pd.DataFrame(records).set_index("parnme")
    
    @staticmethod
    def _validate_parameter_name(name: str) -> None:
        """Validate parameter name."""
        if not name:
            raise ValueError("Parameter name cannot be empty")
        if len(name) > 12:
            raise ValueError(f"Parameter name '{name}' exceeds 12 characters (PEST limit)")
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Parameter name '{name}' contains invalid characters")
    
    @staticmethod
    def _validate_bounds(
        name: str, 
        initial: float, 
        lower: float, 
        upper: float
    ) -> None:
        """Validate parameter bounds."""
        if lower >= upper:
            raise ValueError(f"Parameter '{name}': lower bound must be less than upper bound")
        if not (lower <= initial <= upper):
            raise ValueError(f"Parameter '{name}': initial value must be within bounds")


def get_hspf_hydrology_defaults() -> Dict[str, Dict]:
    """
    Get default HSPF hydrology parameter definitions.
    
    Returns typical parameter ranges for HSPF PERLND hydrology calibration.
    These can be used with ParameterManager.add_from_defaults().
    
    Returns:
        Dictionary of parameter configurations
    """
    return {
        # Infiltration parameters
        "infilt": {
            "initial_value": 0.1,
            "lower_bound": 0.001,
            "upper_bound": 0.5,
            "transform": "log",
            "group": "infiltration",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        # Interflow parameters
        "intfw": {
            "initial_value": 1.0,
            "lower_bound": 0.0,
            "upper_bound": 10.0,
            "transform": "none",
            "group": "interflow",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        "irc": {
            "initial_value": 0.5,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "transform": "none",
            "group": "interflow",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        # Groundwater parameters
        "agwrc": {
            "initial_value": 0.98,
            "lower_bound": 0.85,
            "upper_bound": 0.999,
            "transform": "none",
            "group": "groundwater",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        "deepfr": {
            "initial_value": 0.1,
            "lower_bound": 0.0,
            "upper_bound": 0.5,
            "transform": "none",
            "group": "groundwater",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        "basetp": {
            "initial_value": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "transform": "none",
            "group": "groundwater",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        "agwetp": {
            "initial_value": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "transform": "none",
            "group": "groundwater",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        # Upper zone parameters
        "uzsn": {
            "initial_value": 1.0,
            "lower_bound": 0.05,
            "upper_bound": 10.0,
            "transform": "log",
            "group": "upper_zone",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        "uzrat": {
            "initial_value": 0.5,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "transform": "none",
            "group": "upper_zone",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        # Lower zone parameters
        "lzsn": {
            "initial_value": 5.0,
            "lower_bound": 1.0,
            "upper_bound": 15.0,
            "transform": "none",
            "group": "lower_zone",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
        "lzetp": {
            "initial_value": 0.5,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "transform": "none",
            "group": "lower_zone",
            "block": "PERLND",
            "table": "PWAT-PARM3",
        },
        # Other parameters
        "kvary": {
            "initial_value": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 5.0,
            "transform": "none",
            "group": "groundwater",
            "block": "PERLND",
            "table": "PWAT-PARM2",
        },
    }
