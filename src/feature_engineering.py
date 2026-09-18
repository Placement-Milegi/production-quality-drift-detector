import numpy as np
import pandas as pd

ALIASES = {
    "batch_id": ["batch_id", "batch", "batch_number", "batch_no", "lot_id", "lot_number"],
    "timestamp": ["timestamp", "datetime", "date_time", "date", "recorded_at", "recorded_time"],
    "temperature": ["temperature", "temp", "temp_c", "temperature_c", "temperature_celsius"],
    "pressure": ["pressure", "press", "pressure_psi", "pressure_bar"],
    "cycle_time": ["cycle_time", "cycle_duration", "cycle_time_sec", "duration", "processing_time"],
    "vibration": ["vibration", "vibration_level", "vibration_mm_s", "vibration_rms"],
    "quality_measurement": ["quality_measurement", "quality_score", "quality", "quality_index"],
    "defect_rate": ["defect_rate", "defect_percentage", "defect_pct", "defect_ratio"],
    "temperature_tolerance": ["temperature_tolerance", "temp_tolerance", "temperature_limit", "temp_limit"],
    "pressure_tolerance": ["pressure_tolerance", "pressure_limit", "pressure_upper_limit"],
    "cycle_time_tolerance": ["cycle_time_tolerance", "cycle_tolerance", "cycle_time_limit"],
    "vibration_tolerance": ["vibration_tolerance", "vibration_limit", "vibration_upper_limit"],
}

PROCESS_COLUMNS = ["temperature", "pressure", "cycle_time", "vibration"]

def resolve_columns(df):
    columns = {str(c).lower(): c for c in df.columns}
    resolved = {}
    for concept, aliases in ALIASES.items():
        for alias in aliases:
            if alias.lower() in columns:
                resolved[concept] = columns[alias.lower()]
                break
    return resolved

DEFAULT_TOLERANCES = {
    "temperature": 78.0,
    "pressure": 108.0,
    "cycle_time": 49.0,
    "vibration": 3.3,
}

DEFAULT_BASELINES = {
    "temperature": {"mean": 70.20, "std": 1.47, "median": 70.20},
    "pressure": {"mean": 100.12, "std": 1.85, "median": 100.12},
    "cycle_time": {"mean": 41.90, "std": 1.23, "median": 41.90},
    "vibration": {"mean": 2.01, "std": 0.29, "median": 2.01},
    "quality_measurement": {"mean": 49.94, "std": 0.98, "median": 49.94},
    "defect_rate": {"mean": 0.99, "std": 0.37, "median": 0.99},
}

def create_features(df, window=25, baseline_stats=None):
    result = df.copy()
    resolved = resolve_columns(result)
    stats = baseline_stats or DEFAULT_BASELINES

    timestamp = resolved.get("timestamp")
    batch_id = resolved.get("batch_id")

    if timestamp and pd.api.types.is_datetime64_any_dtype(result[timestamp]):
        result = result.sort_values(timestamp).reset_index(drop=True)
    elif timestamp:
        result[timestamp] = pd.to_datetime(result[timestamp], errors="coerce")
        result = result.sort_values(timestamp).reset_index(drop=True)
    elif batch_id:
        result = result.sort_values(batch_id).reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    result["_sequence"] = np.arange(len(result))

    for concept in PROCESS_COLUMNS:
        column = resolved.get(concept)
        if not column:
            continue

        series = pd.to_numeric(result[column], errors="coerce")
        b_mean = stats.get(concept, {}).get("mean", float(series.mean() if len(series) else 0.0))
        b_std = stats.get(concept, {}).get("std", float(series.std() if len(series) > 1 else 1.0))
        if b_std <= 0:
            b_std = 1.0

        min_p = 1 if len(result) < window else max(2, window // 2)
        rolling_mean = series.rolling(window=window, min_periods=min_p).mean().fillna(b_mean)
        rolling_std = series.rolling(window=window, min_periods=min_p).std().fillna(b_std)
        rolling_std = rolling_std.replace(0, b_std)

        result[f"{concept}_rolling_mean"] = rolling_mean
        result[f"{concept}_rolling_std"] = rolling_std
        result[f"{concept}_rolling_variance"] = series.rolling(window=window, min_periods=min_p).var().fillna(b_std ** 2)
        result[f"{concept}_deviation"] = series - rolling_mean
        result[f"{concept}_rolling_z"] = (series - rolling_mean) / rolling_std
        result[f"{concept}_baseline_z"] = (series - b_mean) / b_std
        result[f"{concept}_change"] = series.diff().fillna(0.0)
        result[f"{concept}_pct_change"] = (series.pct_change() * 100).fillna(0.0)
        
        baseline_roll = series.rolling(window=window, min_periods=min_p).mean().shift(window)
        baseline_roll = baseline_roll.fillna(b_mean)
        result[f"{concept}_baseline_pct_change"] = ((series - baseline_roll) / baseline_roll.replace(0, np.nan)) * 100

    tolerance_map = {
        "temperature": "temperature_tolerance",
        "pressure": "pressure_tolerance",
        "cycle_time": "cycle_time_tolerance",
        "vibration": "vibration_tolerance",
    }

    for concept, tolerance_concept in tolerance_map.items():
        column = resolved.get(concept)
        tolerance = resolved.get(tolerance_concept)
        if column:
            value = pd.to_numeric(result[column], errors="coerce")
            if tolerance and tolerance in result.columns:
                limit = pd.to_numeric(result[tolerance], errors="coerce")
            else:
                limit = DEFAULT_TOLERANCES.get(concept, 100.0)
                
            result[f"{concept}_tolerance_distance"] = limit - value
            result[f"{concept}_tolerance_utilization_pct"] = (value / limit) * 100.0
            result[f"{concept}_near_limit"] = (
                result[f"{concept}_tolerance_utilization_pct"] >= 90.0
            ).astype(int)

    quality = resolved.get("quality_measurement")
    if quality:
        series = pd.to_numeric(result[quality], errors="coerce")
        min_p = 1 if len(result) < window else max(2, window // 2)
        result["quality_change"] = series.diff().fillna(0.0)
        result["quality_pct_change"] = (series.pct_change() * 100).fillna(0.0)
        result["quality_rolling_mean"] = series.rolling(window=window, min_periods=min_p).mean().fillna(series)

    defect = resolved.get("defect_rate")
    if defect:
        series = pd.to_numeric(result[defect], errors="coerce")
        min_p = 1 if len(result) < window else max(2, window // 2)
        result["defect_rate_change"] = series.diff().fillna(0.0)
        result["defect_rate_pct_change"] = (series.pct_change() * 100).fillna(0.0)
        result["defect_rate_rolling_mean"] = series.rolling(window=window, min_periods=min_p).mean().fillna(series)

    return result
