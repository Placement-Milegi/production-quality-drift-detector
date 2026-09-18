import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_cleaning import clean_dataframe
from src.feature_engineering import create_features, DEFAULT_TOLERANCES, DEFAULT_BASELINES
from src.drift_detection import calculate_drift, PROCESS_COLUMNS
from src.evaluation import evaluate_detector, detection_delay

def train_and_evaluate():
    print("=" * 70)
    print("PRODUCTION QUALITY DRIFT DETECTOR — MODEL TRAINING & CALIBRATION")
    print("=" * 70)

    raw_path = ROOT / "Data" / "Data_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    print(f"1. Loading raw data from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    print(f"   Raw shape: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

    print("\n2. Preprocessing & Cleaning data...")
    clean_df = clean_dataframe(raw_df)
    cleaned_path = ROOT / "Data" / "Data_cleaned.csv"
    clean_df.to_csv(cleaned_path, index=False)
    print(f"   Cleaned shape: {clean_df.shape[0]:,} rows × {clean_df.shape[1]} columns")
    print(f"   Saved clean dataset to: {cleaned_path}")

    # Chronological train/test split (80% train / 20% test) to prevent temporal leakage
    train_size = int(len(clean_df) * 0.8)
    train_df = clean_df.iloc[:train_size].copy()
    test_df = clean_df.iloc[train_size:].copy()
    print(f"\n3. Chronological Train/Test Split (80/20):")
    print(f"   Training set: {len(train_df):,} rows (Batches {train_df['batch_id'].min()} to {train_df['batch_id'].max()})")
    print(f"   Test set    : {len(test_df):,} rows (Batches {test_df['batch_id'].min()} to {test_df['batch_id'].max()})")

    # In-control golden baseline calculation from Stable batches in the training period
    stable_train = train_df[train_df["actual_status"] == "Stable"]
    eval_cols = PROCESS_COLUMNS + ["quality_measurement", "defect_rate"]

    baseline_stats = {}
    for col in eval_cols:
        baseline_stats[col] = {
            "mean": float(stable_train[col].mean()),
            "std": float(stable_train[col].std()),
            "median": float(stable_train[col].median()),
            "min": float(stable_train[col].min()),
            "max": float(stable_train[col].max()),
        }
    print("\n4. Computed In-Control Baseline Statistics from Training Data:")
    for col, st in baseline_stats.items():
        print(f"   {col:20}: mean={st['mean']:.3f}, std={st['std']:.3f}, median={st['median']:.3f}")

    # Fit Isolation Forest ML model on clean training process variables
    print("\n5. Fitting Isolation Forest ML Anomaly Model...")
    x_train_if = train_df[PROCESS_COLUMNS].copy()
    iforest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )
    iforest.fit(x_train_if)
    print("   Isolation Forest fitted successfully.")

    # Package Model Artifact
    artifact = {
        "artifact_type": "production_quality_drift_detector",
        "artifact_version": 2.0,
        "model_type": "calibrated_spc_drift_detector_with_isolation_forest",
        "process_columns": PROCESS_COLUMNS,
        "quality_column": "quality_measurement",
        "defect_column": "defect_rate",
        "baseline_stats": baseline_stats,
        "tolerances": DEFAULT_TOLERANCES,
        "thresholds": {
            "z_threshold": 3.0,
            "warning_score": 0.30,
            "critical_score": 0.60,
            "near_limit_pct": 90.0,
            "rolling_window": 25,
            "process_weight": 0.60,
            "quality_defect_weight": 0.40,
        },
        "feature_columns": [
            "temperature_drift_score", "pressure_drift_score",
            "cycle_time_drift_score", "vibration_drift_score",
            "defect_signal", "quality_signal", "process_drift_score"
        ],
        "training_rows": len(train_df),
        "isolation_forest": iforest,
    }

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "drift_detector.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"   Saved updated model artifact to: {model_path}")

    # Generate features and drift detection across the entire dataset
    print("\n6. Running Feature Engineering & Drift Detection pipeline...")
    features_df = create_features(clean_df, window=25, baseline_stats=baseline_stats)
    features_path = ROOT / "Data" / "Data_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"   Features dataset saved: {features_path}")

    drift_df = calculate_drift(
        features_df,
        z_threshold=3.0,
        warning_score=0.30,
        critical_score=0.60,
        baseline_stats=baseline_stats,
        tolerances=DEFAULT_TOLERANCES,
    )

    # Add Isolation Forest predictions to drift results
    x_all = drift_df[PROCESS_COLUMNS].copy()
    drift_df["isolation_forest_anomaly"] = (iforest.predict(x_all) == -1).astype(int)
    drift_df["isolation_forest_score"] = np.round(-iforest.decision_function(x_all), 4)

    drift_results_path = ROOT / "Data" / "Data_drift_results.csv"
    drift_df.to_csv(drift_results_path, index=False)
    print(f"   Drift results saved: {drift_results_path}")

    # 7. Model Evaluation
    print("\n7. Evaluating Model Performance...")
    eval_df, metrics_df = evaluate_detector(drift_df, "actual_status", prediction_column="predicted_drift")

    print("\n--- TEST SET EVALUATION (Held-out 3,000 samples) ---")
    test_eval = drift_df.iloc[train_size:].copy()
    test_y_true = test_eval["actual_status"]
    test_y_pred = test_eval["detected_status"]
    test_acc = accuracy_score(test_y_true, test_y_pred)
    print(f"Test 3-State Accuracy: {test_acc:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(test_y_true, test_y_pred, labels=["Stable", "Warning", "Critical"]))
    print("Test Confusion Matrix:")
    print(confusion_matrix(test_y_true, test_y_pred, labels=["Stable", "Warning", "Critical"]))

    test_y_true_bin = (test_y_true != "Stable").astype(int)
    test_y_pred_bin = (test_y_pred != "Stable").astype(int)
    print("\nTest Binary Drift Detection:")
    print(f"Accuracy : {accuracy_score(test_y_true_bin, test_y_pred_bin):.4f}")
    print(f"Precision: {precision_score(test_y_true_bin, test_y_pred_bin):.4f}")
    print(f"Recall   : {recall_score(test_y_true_bin, test_y_pred_bin):.4f}")
    print(f"F1 Score : {f1_score(test_y_true_bin, test_y_pred_bin):.4f}")

    print("\n--- FULL DATASET EVALUATION (15,000 samples) ---")
    full_y_true = drift_df["actual_status"]
    full_y_pred = drift_df["detected_status"]
    full_acc = accuracy_score(full_y_true, full_y_pred)
    print(f"Overall 3-State Accuracy: {full_acc:.4f}")
    print(classification_report(full_y_true, full_y_pred, labels=["Stable", "Warning", "Critical"]))

    full_y_true_bin = (full_y_true != "Stable").astype(int)
    full_y_pred_bin = (full_y_pred != "Stable").astype(int)
    bin_acc = accuracy_score(full_y_true_bin, full_y_pred_bin)
    bin_prec = precision_score(full_y_true_bin, full_y_pred_bin)
    bin_rec = recall_score(full_y_true_bin, full_y_pred_bin)
    bin_f1 = f1_score(full_y_true_bin, full_y_pred_bin)
    tn, fp, fn, tp = confusion_matrix(full_y_true_bin, full_y_pred_bin, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    delay = detection_delay(eval_df)
    print(f"Detection delay: {delay} observations")

    # Save evaluation reports
    metrics_path = ROOT / "Data" / "drift_evaluation_metrics.csv"
    summary_path = ROOT / "Data" / "drift_evaluation_summary.csv"
    var_report_path = ROOT / "Data" / "drift_variable_report.csv"

    eval_metrics = pd.DataFrame({
        "metric": [
            "accuracy", "precision", "recall", "f1", "false_positive_rate",
            "true_positive", "false_positive", "false_negative", "true_negative"
        ],
        "value": [
            bin_acc, bin_prec, bin_rec, bin_f1, fpr,
            float(tp), float(fp), float(fn), float(tn)
        ]
    })
    eval_metrics.to_csv(metrics_path, index=False)

    eval_summary = pd.DataFrame({
        "evaluation_type": ["ground_truth_status"],
        "ground_truth_available": [True],
        "ground_truth_column": ["actual_status"],
        "detector_output": ["detected_status"],
        "accuracy": [bin_acc],
        "precision": [bin_prec],
        "recall": [bin_rec],
        "f1": [bin_f1],
        "false_positive_rate": [fpr],
        "detection_delay_observations": [delay],
        "three_state_accuracy": [full_acc]
    })
    eval_summary.to_csv(summary_path, index=False)

    var_cols = [f"{c}_drift_score" for c in PROCESS_COLUMNS]
    var_report = pd.DataFrame({
        "variable": PROCESS_COLUMNS,
        "mean_drift_score": [float(drift_df[c].mean()) for c in var_cols],
        "max_drift_score": [float(drift_df[c].max()) for c in var_cols],
        "high_score_observations": [int((drift_df[c] >= 0.60).sum()) for c in var_cols]
    }).sort_values("mean_drift_score", ascending=False)
    var_report.to_csv(var_report_path, index=False)

    print(f"\n   Saved evaluation metrics to: {metrics_path}")
    print(f"   Saved evaluation summary to: {summary_path}")
    print(f"   Saved variable report to: {var_report_path}")
    print("\nTraining and pipeline execution completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
