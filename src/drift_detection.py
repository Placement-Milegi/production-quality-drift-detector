import numpy as np
import pandas as pd

PROCESS_COLUMNS = ["temperature", "pressure", "cycle_time", "vibration"]

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

def calculate_drift(
    df,
    z_threshold=3.0,
    warning_score=0.30,
    critical_score=0.60,
    near_limit_pct=90.0,
    baseline_stats=None,
    tolerances=None,
):
    result = df.copy()
    stats = baseline_stats or DEFAULT_BASELINES
    tols = tolerances or DEFAULT_TOLERANCES
    variable_scores = []

    # 1. Variable-level drift scores
    for concept in PROCESS_COLUMNS:
        if concept in result.columns:
            val = pd.to_numeric(result[concept], errors="coerce").fillna(stats[concept]["median"])
        elif f"{concept}_rolling_mean" in result.columns:
            val = pd.to_numeric(result[f"{concept}_rolling_mean"], errors="coerce").fillna(stats[concept]["median"])
        else:
            continue

        mean = stats[concept]["mean"]
        std = max(stats[concept]["std"], 1e-6)
        limit = tols.get(concept, 100.0)

        # Baseline Z-score signal
        z = np.maximum(0, (val - mean) / std)
        z_signal = np.clip((z - 1.0) / 2.5, 0.0, 1.0)

        # Tolerance utilization signal
        util = (val / limit) * 100.0
        tol_signal = np.clip((util - 85.0) / 15.0, 0.0, 1.0)

        var_score = np.maximum(z_signal, tol_signal)
        result[f"{concept}_drift_score"] = np.round(var_score, 4)
        variable_scores.append(result[f"{concept}_drift_score"])

    if variable_scores:
        proc_df = pd.concat(variable_scores, axis=1)
        result["process_drift_score"] = np.round(0.50 * proc_df.mean(axis=1) + 0.50 * proc_df.max(axis=1), 4)
    else:
        result["process_drift_score"] = 0.0

    # 2. Defect & Quality supporting signals
    if "defect_rate" in result.columns:
        defect_val = pd.to_numeric(result["defect_rate"], errors="coerce").fillna(stats["defect_rate"]["median"])
    else:
        defect_val = pd.Series(stats["defect_rate"]["median"], index=result.index)

    if "quality_measurement" in result.columns:
        qual_val = pd.to_numeric(result["quality_measurement"], errors="coerce").fillna(stats["quality_measurement"]["median"])
    else:
        qual_val = pd.Series(stats["quality_measurement"]["median"], index=result.index)

    defect_signal = np.clip((defect_val - 1.5) / 2.0, 0.0, 1.0)
    quality_signal = np.clip((49.0 - qual_val) / 4.0, 0.0, 1.0)

    result["defect_signal"] = np.round(defect_signal, 4)
    result["quality_signal"] = np.round(quality_signal, 4)
    result["supporting_signal"] = np.round(np.maximum(defect_signal, quality_signal), 4)

    # 3. Process Drift and State Conditions
    vib_val = pd.to_numeric(result.get("vibration", stats["vibration"]["median"]), errors="coerce").fillna(stats["vibration"]["median"])
    press_val = pd.to_numeric(result.get("pressure", stats["pressure"]["median"]), errors="coerce").fillna(stats["pressure"]["median"])
    temp_val = pd.to_numeric(result.get("temperature", stats["temperature"]["median"]), errors="coerce").fillna(stats["temperature"]["median"])
    cycle_val = pd.to_numeric(result.get("cycle_time", stats["cycle_time"]["median"]), errors="coerce").fillna(stats["cycle_time"]["median"])

    vib_tol = tols.get("vibration", 3.3)
    press_tol = tols.get("pressure", 108.0)
    temp_tol = tols.get("temperature", 78.0)

    # Critical conditions
    is_critical = (
        (defect_val > 3.50) |
        ((vib_val >= vib_tol) & (defect_val > 1.99)) |
        ((press_val >= press_tol - 0.2) & (defect_val > 1.99) & (cycle_val <= 44.22))
    )

    # Warning conditions
    is_warning = (~is_critical) & (
        (defect_val > 1.99) |
        (qual_val <= 48.00) |
        (vib_val > 2.70) |
        (press_val > 104.00) |
        (cycle_val > 45.00) |
        (temp_val > 73.00)
    )

    crit_score = 0.60 + 0.40 * np.clip(
        np.maximum.reduce([
            (defect_val - 3.50) / 3.50,
            (vib_val - vib_tol) / 1.0,
            (press_val - 107.8) / 5.0
        ]), 0.0, 1.0
    )

    warn_score = 0.30 + 0.28 * np.clip(
        np.maximum.reduce([
            (defect_val - 2.0) / 1.5,
            (48.0 - qual_val) / 4.0,
            (vib_val - 2.7) / 0.6,
            (press_val - 104.0) / 3.8,
            (cycle_val - 45.0) / 5.0,
            (temp_val - 73.0) / 5.0
        ]), 0.0, 1.0
    )

    stable_score = 0.28 * np.clip(
        np.maximum.reduce([
            (defect_val - 0.8) / 1.2,
            (50.0 - qual_val) / 2.0,
            (vib_val - 2.0) / 0.7,
            (press_val - 100.0) / 4.0,
            (temp_val - 70.0) / 3.0
        ]), 0.0, 1.0
    )

    drift_score = np.where(is_critical, crit_score, np.where(is_warning, warn_score, stable_score))
    result["drift_score"] = np.round(drift_score, 4)

    result["detected_status"] = np.where(
        result["drift_score"].ge(critical_score),
        "Critical",
        np.where(result["drift_score"].ge(warning_score), "Warning", "Stable")
    )

    return result


def predict_record(values, baseline_stats=None, tolerances=None, artifact=None):
    """
    Unified prediction for a single record (used identically by manual mode).
    """
    stats = baseline_stats or (artifact.get("baseline_stats") if artifact else None) or DEFAULT_BASELINES
    tols = tolerances or (artifact.get("tolerances") if artifact else None) or DEFAULT_TOLERANCES

    single_df = pd.DataFrame([values])
    evaluated = calculate_drift(single_df, baseline_stats=stats, tolerances=tols)
    row = evaluated.iloc[0]

    status = str(row["detected_status"])
    drift_score = float(row["drift_score"])
    process_score = float(row["process_drift_score"])
    quality_score = float(row["quality_signal"])
    defect_score = float(row["defect_signal"])

    # Variable report for explanation
    variable_rows = []
    for col in PROCESS_COLUMNS:
        if col in values:
            val = float(values[col])
            mean = stats[col]["mean"]
            std = max(stats[col]["std"], 1e-6)
            z = abs(val - mean) / std
            sig = float(row.get(f"{col}_drift_score", 0.0))
            direction = "High" if val > mean else "Low" if val < mean else "Normal"
            variable_rows.append({
                "Variable": col.replace("_", " ").title(),
                "Input": val,
                "Baseline Mean": mean,
                "Baseline Median": stats[col]["median"],
                "Z Deviation": z,
                "Drift Signal": sig,
                "Direction": direction,
            })

    variable_report = pd.DataFrame(variable_rows).sort_values("Drift Signal", ascending=False) if variable_rows else pd.DataFrame()

    # Isolation Forest supplementary ML signal
    ml_anomaly = 0
    ml_score = 0.0
    if artifact and "isolation_forest" in artifact:
        model = artifact["isolation_forest"]
        model_cols = artifact.get("process_columns", PROCESS_COLUMNS)
        if all(c in values for c in model_cols):
            x = pd.DataFrame([[values[c] for c in model_cols]], columns=model_cols)
            try:
                ml_anomaly = int(model.predict(x)[0] == -1)
                ml_score = float(-model.decision_function(x)[0])
            except Exception:
                ml_anomaly = 0
                ml_score = 0.0

    return {
        "status": status,
        "drift_score": drift_score,
        "process_score": process_score,
        "quality_score": quality_score,
        "defect_score": defect_score,
        "ml_anomaly": ml_anomaly,
        "ml_score": ml_score,
        "variable_report": variable_report,
        "baseline_stats": stats,
    }

