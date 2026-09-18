# Production Quality Drift Detector

A data-driven manufacturing process monitoring system designed to detect gradual production-quality drift before it becomes a major defect problem.

## Problem Statement

Production lines can gradually drift toward failure even while individual parts continue to pass tolerance checks. The objective of this project is to monitor production measurements, identify abnormal changes and drift, determine which variables changed most, and communicate the current process condition as:

- Stable
- Warning
- Critical

The system is designed for process-monitoring demonstration and does not replace calibrated industrial quality systems.

## Project Workflow

```text
Raw Manufacturing Data
        |
        v
01 Data Exploration
EDA + Data Cleaning
        |
        v
Data_cleaned.csv
        |
        v
02 Feature Engineering
Rolling Statistics
Z-Scores
Change Features
Tolerance Features
Quality / Defect Features
        |
        v
Data_features.csv
        |
        v
03 Drift Detection
Statistical Signals
Tolerance Signals
Directional Drift
Drift Score
Stable / Warning / Critical
Optional Isolation Forest
        |
        v
Data_drift_results.csv
        |
        v
04 Model Evaluation
Ground Truth
Precision / Recall / F1
Confusion Matrix
Detection Delay
        |
        v
Evaluation Reports
```

## Repository Structure

```text
Hackathon/
|
├── .venv/
|
├── Data/
│   ├── Data_raw.csv
│   ├── Data_cleaned.csv
│   ├── Data_features.csv
│   ├── Data_drift_results.csv
│   ├── drift_evaluation_metrics.csv
│   ├── drift_variable_report.csv
│   └── drift_evaluation_summary.csv
|
├── Notebook/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Drift_Detection.ipynb
│   └── 04_Model_Evaluation.ipynb
|
├── requirements.txt
├── README.md
└── LICENSE
```

Generated CSV files are stored in the same `Data/` directory as the raw dataset.

## Notebook Pipeline

### 01 — Data Exploration

Performs:

- Dataset discovery
- Dataset structure inspection
- Data types and missing-value analysis
- Duplicate analysis
- Numeric and categorical exploration
- Outlier inspection
- Manufacturing-specific exploratory analysis
- Data cleaning
- Cleaning validation
- Export of `Data_cleaned.csv`

The notebook is designed to avoid depending on one fixed dataset filename.

### 02 — Feature Engineering

Creates monitoring features from the cleaned dataset.

Main feature groups:

- Rolling mean
- Rolling standard deviation
- Rolling variance
- Deviation from rolling baseline
- Rolling z-score
- Consecutive change
- Percentage change
- Baseline percentage change
- Tolerance distance
- Tolerance utilization
- Near-limit indicators
- Quality change
- Defect-rate change
- Rolling quality and defect metrics

Output:

```text
Data/Data_features.csv
```

### 03 — Drift Detection

Uses engineered features to produce process-drift evidence.

Detection signals include:

- Rolling z-score threshold
- Tolerance proximity
- Directional baseline shift
- Quality movement
- Defect-rate movement
- Variable-level drift scores
- Overall drift score

Operational status:

```text
Stable
Warning
Critical
```

An optional Isolation Forest detector can provide an additional anomaly signal when suitable numeric process variables are available.

Output:

```text
Data/Data_drift_results.csv
```

### 04 — Model Evaluation

Evaluates the detector when a ground-truth drift label is available.

Metrics include:

- Accuracy
- Precision
- Recall
- F1 score
- False-positive rate
- Confusion matrix
- Detection delay
- Variable-level drift evidence

If the dataset does not contain ground-truth labels, the notebook does not invent them. It reports that supervised evaluation is unavailable.

Outputs:

```text
Data/drift_evaluation_metrics.csv
Data/drift_variable_report.csv
Data/drift_evaluation_summary.csv
```

## Expected Manufacturing Concepts

The notebooks automatically attempt to identify common versions of these concepts:

| Concept | Examples |
|---|---|
| Batch | `batch_id`, `batch`, `batch_number` |
| Time | `timestamp`, `datetime`, `recorded_at` |
| Temperature | `temperature`, `temp`, `temperature_c` |
| Pressure | `pressure`, `pressure_psi`, `pressure_bar` |
| Cycle time | `cycle_time`, `cycle_duration`, `processing_time` |
| Vibration | `vibration`, `vibration_level`, `vibration_rms` |
| Quality | `quality_measurement`, `quality_score`, `quality` |
| Defect rate | `defect_rate`, `defect_percentage`, `defect_pct` |
| Tolerance | Variable-specific tolerance or limit columns |

The exact dataset schema may vary. Column detection is therefore handled programmatically where possible.

## Installation

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

## Running the Project

Place the raw manufacturing dataset in:

```text
Data/
```

The recommended file is:

```text
Data/Data_raw.csv
```

Open:

```text
Notebook/01_Data_Exploration.ipynb
```

Run the notebook from top to bottom.

Then run:

```text
Notebook/02_Feature_Engineering.ipynb
Notebook/03_Drift_Detection.ipynb
Notebook/04_Model_Evaluation.ipynb
```

Each stage produces the next stage's input inside the `Data/` directory.

## PS Alignment

The project addresses the requested components:

| PS Requirement | Implementation |
|---|---|
| Load process measurements | Data loading and automatic column detection |
| Rolling mean | Feature Engineering |
| Rolling variance | Feature Engineering |
| Tolerance distance | Feature Engineering |
| Defect rate | Quality / defect features |
| Statistical drift detection | Z-score and directional signals |
| Anomaly detection | Optional Isolation Forest |
| Stable / Warning / Critical | Drift Detection |
| Identify changed variables | Variable-level drift scores |
| Dataset analysis | Data Exploration |
| Drift score/status by batch | Drift Detection |
| Visual timeline | Drift Detection / Evaluation |
| Detect scripted drift | Evaluation when ground truth is available |
| Compare drift methods | Extensible detection architecture |
| Automated incident reporting | Future application layer |

## Design Principles

### Reusable

The notebooks do not require one exact raw CSV filename and attempt to detect common column-name variants.

### Reproducible

Processing is separated into exploration, feature engineering, detection and evaluation stages.

### Traceable

The raw dataset remains separate from generated datasets.

### Evidence-Based

The detector exposes the signals contributing to drift rather than hiding the process behind one black-box prediction.

### Safe Data Handling

The raw dataset is not overwritten by the cleaning pipeline.

## Current Limitations

- The project is a prototype for process-monitoring demonstration.
- Detection thresholds are configurable and should be calibrated using real production data before operational deployment.
- A ground-truth drift label is required for supervised performance metrics.
- Isolation Forest is optional and should be interpreted as supporting anomaly evidence.
- Tolerance analysis depends on appropriate tolerance or limit information being available in the dataset.
- The system does not replace certified industrial quality-control, calibration or safety systems.

## Future Extensions

- Interactive monitoring dashboard
- Automated incident reports
- Batch-level aggregation
- Estimated batches until tolerance failure
- Comparison of multiple statistical drift methods
- Alerting and notification system
- Model/version tracking
- Production database integration
- Real-time streaming monitoring

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## License

This project is released under the MIT License. See `LICENSE` for details.
