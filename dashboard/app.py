from pathlib import Path
import io
import pickle
import sys
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_cleaning import clean_dataframe
from src.feature_engineering import create_features, DEFAULT_TOLERANCES, DEFAULT_BASELINES
from src.drift_detection import calculate_drift, predict_record, PROCESS_COLUMNS

st.set_page_config(
    page_title="Production Quality Drift Detector",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.hero {padding: 1.25rem 1.5rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px; margin-bottom: 1rem;}
.hero h1 {margin: 0 0 .25rem 0;}
.hero p {margin: 0; opacity: .72;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1rem; min-height: 105px;}
.label {font-size: .82rem; opacity: .65;}
.value {font-size: 1.55rem; font-weight: 700; margin: .2rem 0;}
.prediction {padding: 1rem 1.25rem; border: 2px solid rgba(128,128,128,.35); border-radius: 16px; margin: .5rem 0 1rem 0;}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, caption=""):
    st.markdown(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="label">{caption}</div></div>',
        unsafe_allow_html=True,
    )


def locate(*path_names):
    candidates = []
    for path_name in path_names:
        candidates.extend([
            ROOT / path_name,
            Path(__file__).resolve().parent / path_name,
        ])
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_artifact(path):
    with open(path, "rb") as file:
        return pickle.load(file)


@st.cache_data
def read_csv_bytes(data):
    return pd.read_csv(io.BytesIO(data))


@st.cache_data
def process_data(data, z_threshold, warning_score, critical_score, rolling_window):
    raw = pd.read_csv(io.BytesIO(data))
    cleaned = clean_dataframe(raw)
    features = create_features(cleaned, window=rolling_window, baseline_stats=baseline_stats)
    results = calculate_drift(
        features,
        z_threshold=z_threshold,
        warning_score=warning_score,
        critical_score=critical_score,
        baseline_stats=baseline_stats,
        tolerances=tolerances,
    )
    return cleaned, results


def add_ml_signal(results, artifact):
    model = artifact.get("isolation_forest")
    process_cols = artifact.get("process_columns", PROCESS_COLUMNS)
    results = results.copy()

    if model is None or not all(c in results.columns for c in process_cols):
        results["ml_anomaly"] = 0
        results["ml_anomaly_score"] = 0.0
        return results

    x = results[process_cols].apply(pd.to_numeric, errors="coerce")
    valid = x.notna().all(axis=1)
    results["ml_anomaly"] = 0
    results["ml_anomaly_score"] = 0.0

    if valid.any():
        try:
            results.loc[valid, "ml_anomaly"] = (model.predict(x.loc[valid]) == -1).astype(int)
            results.loc[valid, "ml_anomaly_score"] = np.round(-model.decision_function(x.loc[valid]), 4)
        except Exception:
            pass

    return results


def calculate_evaluation(df):
    if "detected_status" not in df.columns:
        return None

    candidates = [
        "actual_status", "actual_drift", "drift_label", "ground_truth",
        "scripted_drift", "is_drift", "process_status", "quality_status", "condition"
    ]
    truth_col = next((c for c in candidates if c in df.columns and c != "detected_status"), None)
    if truth_col is None:
        return None

    drift_words = {"warning", "critical", "drift", "1", "true", "yes", "failed", "failure", "defect", "defective"}
    normal_words = {"stable", "normal", "no drift", "0", "false", "no", "accepted", "pass", "passed", "good", "ok"}
    truth_raw = df[truth_col].astype(str).str.strip().str.lower()
    pred_raw = df["detected_status"].astype(str).str.strip().str.lower()
    truth = truth_raw.map(lambda x: 1 if x in drift_words else (0 if x in normal_words else np.nan))
    pred = pred_raw.map(lambda x: 1 if x in drift_words else (0 if x in normal_words else np.nan))
    valid = truth.notna() & pred.notna()
    if not valid.any():
        return None

    y_true = truth[valid].astype(int)
    y_pred = pred[valid].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Also 3-state accuracy if categorical
    raw_valid = df[[truth_col, "detected_status"]].dropna()
    three_state_acc = (raw_valid[truth_col].astype(str).str.title() == raw_valid["detected_status"].astype(str).str.title()).mean()

    return {
        "truth_col": truth_col,
        "accuracy": accuracy_score(y_true, y_pred),
        "three_state_accuracy": float(three_state_acc),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# Load trained model artifact
artifact_path = locate("models/drift_detector.pkl", "Model/drift_detector.pkl")
if artifact_path is None:
    st.error("drift_detector.pkl was not found. Place it inside models/ or run src/train.py.")
    st.stop()

artifact = load_artifact(str(artifact_path))
thresholds = artifact.get("thresholds", {})
z_threshold = float(thresholds.get("z_threshold", 3.0))
warning_score = float(thresholds.get("warning_score", 0.30))
critical_score = float(thresholds.get("critical_score", 0.60))
rolling_window = int(thresholds.get("rolling_window", 25))
process_columns = artifact.get("process_columns", PROCESS_COLUMNS)
quality_column = artifact.get("quality_column", "quality_measurement")
defect_column = artifact.get("defect_column", "defect_rate")
baseline_stats = artifact.get("baseline_stats", DEFAULT_BASELINES)
tolerances = artifact.get("tolerances", DEFAULT_TOLERANCES)

st.markdown("""
<div class="hero">
<h1>🏭 Production Quality Drift Detector</h1>
<p>Monitor manufacturing production conditions, detect process quality drift, identify contributing variables, and view validated model metrics.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Prediction Mode")
    mode = st.radio(
        "Select Operation Mode",
        ["CSV Upload Prediction", "Manual Single-Record Prediction"],
    )
    st.divider()
    st.header("Detector Specification")
    st.caption(f"**Model**: {artifact.get('model_type', 'Drift Detector')}")
    st.caption(f"**Training Observations**: {artifact.get('training_rows', 'N/A'):,}")
    st.caption(f"**Warning Threshold**: {warning_score:.2f}")
    st.caption(f"**Critical Threshold**: {critical_score:.2f}")

# ==========================================
# MODE 1: CSV Upload Prediction
# ==========================================
if mode == "CSV Upload Prediction":
    st.subheader("CSV Batch Prediction")
    st.caption("Upload a production CSV file to evaluate drift, detect anomalous runs, and review metrics.")

    default_raw = locate("Data/Data_raw.csv")
    upload = st.file_uploader("Upload production CSV", type=["csv"])

    if upload is not None:
        raw_bytes = upload.getvalue()
        source_name = upload.name
    elif default_raw is not None:
        raw_bytes = default_raw.read_bytes()
        source_name = default_raw.name
        st.info(f"Using default sample dataset (`{source_name}`) for demonstration. Upload a file above to test your own data.")
    else:
        st.warning("Please upload a production CSV file to begin analysis.")
        st.stop()

    total_rows = len(read_csv_bytes(raw_bytes))

    try:
        with st.spinner("Executing data cleaning, feature extraction, and drift prediction pipeline..."):
            cleaned_df, results = process_data(
                raw_bytes,
                z_threshold,
                warning_score,
                critical_score,
                rolling_window,
            )
            results = add_ml_signal(results, artifact)
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        st.stop()

    if results.empty:
        st.warning("No valid observations were produced from the file.")
        st.stop()

    latest = results.iloc[-1]
    current_status = str(latest.get("detected_status", "Unknown"))
    current_score = float(pd.to_numeric(pd.Series([latest.get("drift_score", np.nan)]), errors="coerce").iloc[0])

    # Key metrics display
    st.subheader("Current Production Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Current State", current_status, "Latest processed observation")
    with c2:
        metric_card("Drift Score", f"{current_score:.3f}", "0 = Stable, ≥0.3 = Warning, ≥0.6 = Critical")
    with c3:
        metric_card("Processed Rows", f"{len(results):,}", f"Cleaned observations")
    with c4:
        anomaly_count = int((results["ml_anomaly"] == 1).sum()) if "ml_anomaly" in results else 0
        metric_card("ML Anomalies", f"{anomaly_count:,}", "Isolation Forest flags")

    # Drift Score Timeline
    st.subheader("Process Drift Timeline")
    timeline = results.copy()
    if "timestamp" in timeline.columns:
        timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
        timeline = timeline.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    else:
        timeline.index = np.arange(len(timeline))
    st.line_chart(timeline[["drift_score"]], height=300)

    # Variable-level breakdown
    st.subheader("Variable Contribution Analysis")
    variable_cols = [c for c in results.columns if c.endswith("_drift_score") and c not in {"process_drift_score", "drift_score"}]
    if variable_cols:
        variable_report = pd.DataFrame({
            "Variable": [c.replace("_drift_score", "").replace("_", " ").title() for c in variable_cols],
            "Latest Drift Score": [pd.to_numeric(pd.Series([latest[c]]), errors="coerce").iloc[0] for c in variable_cols],
            "Average Drift Score": [pd.to_numeric(results[c], errors="coerce").mean() for c in variable_cols],
            "High Drift Count (≥0.60)": [int((pd.to_numeric(results[c], errors="coerce") >= critical_score).sum()) for c in variable_cols],
        }).sort_values("Latest Drift Score", ascending=False)
        st.dataframe(
            variable_report.style.format({"Latest Drift Score": "{:.3f}", "Average Drift Score": "{:.3f}"}),
            hide_index=True,
            use_container_width=True,
        )

    # Model Evaluation (if ground truth column exists)
    evaluation = calculate_evaluation(results)
    if evaluation:
        st.subheader("Model Evaluation Performance")
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            metric_card("3-State Accuracy", f"{evaluation['three_state_accuracy']:.1%}", "Stable / Warning / Critical")
        with e2:
            metric_card("Drift Precision", f"{evaluation['precision']:.1%}", "Warning + Critical")
        with e3:
            metric_card("Drift Recall", f"{evaluation['recall']:.1%}", "Detection sensitivity")
        with e4:
            metric_card("Drift F1-Score", f"{evaluation['f1']:.1%}", "Harmonic mean")
        with e5:
            metric_card("False Alarm Rate", f"{evaluation['fpr']:.1%}", "False positive rate")

        st.caption(
            f"Ground-truth reference column: `{evaluation['truth_col']}` • "
            f"True Positives: {evaluation['tp']:,} • True Negatives: {evaluation['tn']:,} • "
            f"False Positives: {evaluation['fp']:,} • False Negatives: {evaluation['fn']:,}"
        )
        cm = pd.DataFrame(
            [[evaluation["tn"], evaluation["fp"]], [evaluation["fn"], evaluation["tp"]]],
            index=["Actual Stable", "Actual Drift (Warn/Crit)"],
            columns=["Predicted Stable", "Predicted Drift (Warn/Crit)"],
        )
        st.dataframe(cm, use_container_width=True)

    # Cleaned Data CSV Download Artifact
    st.subheader("Export Cleaned Data")
    st.caption("Download the preprocessed, deduplicated, and range-validated dataset.")
    st.download_button(
        "📥 Download Cleaned Data (CSV)",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="Data_cleaned.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ==========================================
# MODE 2: Manual Single-Record Prediction
# ==========================================
else:
    st.subheader("Manual Single-Record Prediction")
    st.caption("Enter a single production reading to evaluate drift severity and operational state using the unified detector engine.")

    input_columns = process_columns + [quality_column, defect_column]
    defaults = {col: baseline_stats.get(col, {}).get("median", 0.0) for col in input_columns}

    with st.form("manual_single_record_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            temp_in = st.number_input("Temperature (°C)", value=float(defaults.get("temperature", 70.2)), format="%.3f")
        with c2:
            press_in = st.number_input("Pressure (psi)", value=float(defaults.get("pressure", 100.1)), format="%.3f")
        with c3:
            cycle_in = st.number_input("Cycle Time (sec)", value=float(defaults.get("cycle_time", 41.9)), format="%.3f")
        with c4:
            vib_in = st.number_input("Vibration (mm/s)", value=float(defaults.get("vibration", 2.0)), format="%.3f")

        c5, c6 = st.columns(2)
        with c5:
            qual_in = st.number_input("Quality Measurement", value=float(defaults.get(quality_column, 49.9)), format="%.3f")
        with c6:
            defect_in = st.number_input("Defect Rate (%)", value=float(defaults.get(defect_column, 0.98)), format="%.3f")

        submitted = st.form_submit_button("🔍 Predict Operational State", use_container_width=True)

    if submitted:
        input_values = {
            "temperature": temp_in,
            "pressure": press_in,
            "cycle_time": cycle_in,
            "vibration": vib_in,
            quality_column: qual_in,
            defect_column: defect_in,
        }

        try:
            prediction = predict_record(
                input_values,
                baseline_stats=baseline_stats,
                tolerances=tolerances,
                artifact=artifact,
            )

            current_status = prediction["status"]
            current_score = prediction["drift_score"]
            ml_anomaly = prediction["ml_anomaly"]

            st.markdown(
                f'<div class="prediction"><div class="label">PREDICTED PRODUCTION STATE</div><div class="value">{current_status}</div>'
                f'<div class="label">Drift score: {current_score:.3f} • Isolation Forest anomaly: {"Yes" if ml_anomaly else "No"}</div></div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Predicted State", current_status, "Stable / Warning / Critical")
            with m2:
                metric_card("Drift Score", f"{current_score:.3f}", "0 = Stable, ≥0.3 = Warning, ≥0.6 = Critical")
            with m3:
                metric_card("ML Anomaly", "Flagged" if ml_anomaly else "Normal", "Isolation Forest signal")

            st.subheader("Root-Cause Explanation")
            report = prediction["variable_report"]
            if not report.empty:
                st.dataframe(
                    report.style.format({
                        "Input": "{:.3f}",
                        "Baseline Mean": "{:.3f}",
                        "Baseline Median": "{:.3f}",
                        "Z Deviation": "{:.2f}",
                        "Drift Signal": "{:.3f}",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

            q1, q2, q3 = st.columns(3)
            with q1:
                metric_card("Process Drift Signal", f'{prediction["process_score"]:.3f}', "Process variable deviation")
            with q2:
                metric_card("Quality Degradation", f'{prediction["quality_score"]:.3f}', "Quality measurement reduction")
            with q3:
                metric_card("Defect Rate Elevation", f'{prediction["defect_score"]:.3f}', "Defect rate rise")

        except Exception as exc:
            st.error(f"Prediction error: {exc}")

    # Reference Baseline Table
    st.subheader("Reference In-Control Baseline")
    baseline_view = pd.DataFrame([
        {
            "Variable": c.replace("_", " ").title(),
            "Baseline Median": baseline_stats[c]["median"],
            "Baseline Mean": baseline_stats[c]["mean"],
            "Std Dev": baseline_stats[c]["std"],
            "Observed Min": baseline_stats[c].get("min", np.nan),
            "Observed Max": baseline_stats[c].get("max", np.nan),
        }
        for c in input_columns if c in baseline_stats
    ])
    st.dataframe(
        baseline_view.style.format({
            "Baseline Median": "{:.3f}", "Baseline Mean": "{:.3f}", "Std Dev": "{:.3f}",
            "Observed Min": "{:.3f}", "Observed Max": "{:.3f}"
        }),
        hide_index=True,
        use_container_width=True,
    )
