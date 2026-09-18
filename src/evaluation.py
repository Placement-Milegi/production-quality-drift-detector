import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def to_binary_labels(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() >= 0.90:
        return (numeric > 0).astype(int)

    positive = {
        "1", "true", "yes", "y", "drift",
        "anomaly", "abnormal", "critical", "warning"
    }

    values = series.astype("string").str.strip().str.lower()
    return values.isin(positive).astype(int)

def evaluate_detector(df, ground_truth_column, prediction_column="predicted_drift"):
    if ground_truth_column not in df.columns:
        raise ValueError(f"Ground-truth column '{ground_truth_column}' was not found.")

    result = df.copy()

    result["actual_drift"] = to_binary_labels(result[ground_truth_column])

    if prediction_column not in result.columns:
        if "detected_status" in result.columns:
            result[prediction_column] = (
                result["detected_status"]
                .astype("string")
                .str.lower()
                .isin(["warning", "critical"])
                .astype(int)
            )
        elif "drift_score" in result.columns:
            result[prediction_column] = (
                result["drift_score"] >= 0.30
            ).astype(int)
        else:
            raise ValueError("No detector output was found.")

    y_true = result["actual_drift"]
    y_pred = result[prediction_column]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn
    }

    return result, pd.DataFrame({
        "metric": list(metrics.keys()),
        "value": list(metrics.values())
    })

def detection_delay(df, actual_column="actual_drift", predicted_column="predicted_drift"):
    actual_indices = df.index[df[actual_column] == 1].tolist()
    predicted_indices = df.index[df[predicted_column] == 1].tolist()

    if not actual_indices:
        return np.nan

    onset = actual_indices[0]
    later = [index for index in predicted_indices if index >= onset]

    if not later:
        return np.nan

    return later[0] - onset
