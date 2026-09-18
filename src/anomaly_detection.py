import numpy as np
import pandas as pd

def run_isolation_forest(df, columns, contamination="auto", n_estimators=200, random_state=42):
    from sklearn.ensemble import IsolationForest

    usable = [
        column for column in columns
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not usable:
        raise ValueError("No suitable numeric columns were provided.")

    result = df.copy()
    model_data = result[usable].replace([np.inf, -np.inf], np.nan)
    model_data = model_data.fillna(model_data.median())

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )

    predictions = model.fit_predict(model_data)
    scores = model.decision_function(model_data)

    result["isolation_forest_anomaly"] = (predictions == -1).astype(int)
    result["isolation_forest_score"] = -scores

    return result, model
