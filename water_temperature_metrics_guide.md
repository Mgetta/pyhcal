# Water Temperature Metrics

This module provides quantitative metrics for water temperature calibration, specifically designed to analyze diurnal (daily) cycles in hourly timeseries data.

## Overview

Water temperature follows a strong diurnal cycle due to solar heating during the day and cooling at night. This module fits a statistical sinusoidal model to hourly temperature data to extract:

1. **Amplitude**: The magnitude of temperature variation (°C)
2. **Phase**: The time of day when maximum temperature occurs (hours, 0-24)
3. **Trend**: Low-frequency temperature trend over time (°C per day)
4. **Mean Temperature**: The baseline temperature around which the diurnal cycle oscillates (°C)

## Functions

### `fit_diurnal_cycle(hourly_series)`

Fits a sinusoidal model to hourly water temperature data.

**Model:** `T(t) = A * sin(2π/24 * (t - φ)) + trend*t + offset`

**Parameters:**
- `hourly_series`: pandas.Series with datetime index containing hourly temperature data

**Returns:** Dictionary with:
- `amplitude`: Amplitude of diurnal cycle (°C)
- `phase_hours`: Time of maximum temperature (0-24 hours)
- `trend_per_day`: Temperature trend (°C per day)
- `mean_temp`: Baseline temperature around which the diurnal cycle oscillates (°C)
- `r_squared`: Goodness of fit

### `daily_temperature_stats(hourly_series)`

Calculates daily statistics from hourly temperature data.

**Parameters:**
- `hourly_series`: pandas.Series with datetime index containing hourly temperature data

**Returns:** DataFrame with daily values:
- `daily_mean`: Daily mean temperature
- `daily_min`: Daily minimum temperature
- `daily_max`: Daily maximum temperature
- `daily_range`: Daily temperature range (max - min)
- `time_of_min`: Hour when minimum occurs
- `time_of_max`: Hour when maximum occurs

### `temperature_diurnal_metrics(df_hourly)`

Calculates comprehensive diurnal cycle metrics for both observed and simulated data.

**Parameters:**
- `df_hourly`: DataFrame with columns 'observed' and 'simulated' (hourly data in °C)

**Returns:** DataFrame comparing diurnal metrics between observed and simulated data, including absolute and percentage errors.

### `temperature_stats(df_hourly)`

Calculates comprehensive temperature statistics combining diurnal metrics with traditional aggregated statistics.

**Parameters:**
- `df_hourly`: DataFrame with columns 'observed' and 'simulated' (hourly data in °C)

**Returns:** DataFrame with diurnal characteristics plus monthly and seasonal statistics.

## Example Usage

```python
import pandas as pd
from pyhcal import metrics

# Load hourly water temperature data
# df_hourly should have 'observed' and 'simulated' columns
# with a datetime index

# Get comprehensive temperature statistics
temp_stats = metrics.temperature_stats(df_hourly)
print(temp_stats)

# Get just diurnal metrics
diurnal_metrics = metrics.temperature_diurnal_metrics(df_hourly)
print(f"Observed amplitude: {diurnal_metrics.loc['amplitude', 'observed']:.2f}°C")
print(f"Simulated amplitude: {diurnal_metrics.loc['amplitude', 'simulated']:.2f}°C")
print(f"Time of peak (observed): {diurnal_metrics.loc['phase_hours', 'observed']:.1f} hours")
print(f"Time of peak (simulated): {diurnal_metrics.loc['phase_hours', 'simulated']:.1f} hours")

# Fit diurnal cycle for individual series
obs_fit = metrics.fit_diurnal_cycle(df_hourly['observed'])
print(f"R-squared: {obs_fit['r_squared']:.3f}")
print(f"Trend: {obs_fit['trend_per_day']:.4f}°C/day")

# Get daily statistics
daily_stats = metrics.daily_temperature_stats(df_hourly['observed'])
print(daily_stats.head())
```

## Integration with Existing Workflow

These functions follow the same pattern as existing metrics in the module:
- Accept DataFrames with 'observed' and 'simulated' columns
- Return standardized DataFrames with metrics
- Compatible with the 'degC' unit type already used in the `aggregate()`, `total()`, `annual()`, `monthly()`, and `season()` functions

## Notes

- The sinusoidal fit assumes a 24-hour period
- Works best with at least 24 hours of continuous data
- Returns NaN values if fitting fails or insufficient data is provided
- The model captures the dominant daily cycle and a linear trend, suitable for typical water temperature patterns
