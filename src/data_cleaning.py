import re
import numpy as np
import pandas as pd

def normalize_column_name(name):
    value = str(name).strip().lower()
    value = re.sub(r"[%/()\-]+", "_", value)
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")

def clean_column_names(df):
    result = df.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result

def convert_numeric_like_columns(df, threshold=0.90):
    result = df.copy()
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            continue
        if any(term in str(column).lower() for term in ["time", "date", "id", "batch"]):
            continue
        converted = pd.to_numeric(
            result[column].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
            errors="coerce"
        )
        if converted.notna().mean() >= threshold:
            result[column] = converted
    return result

def clean_categorical_values(df):
    result = df.copy()
    for col in result.select_dtypes(include=["object", "string"]).columns:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ["time", "date"]):
            continue
        series = result[col].astype(str).str.strip()
        if "machine" in col_lower:
            result[col] = series.str.upper()
        elif any(term in col_lower for term in ["product_line", "line", "shift", "status", "outcome"]):
            result[col] = series.str.title()
        else:
            result[col] = series.replace("nan", np.nan)
    return result

def validate_ranges(df):
    result = df.copy()
    if "defect_rate" in result.columns:
        result["defect_rate"] = result["defect_rate"].clip(lower=0.0)
    if "maintenance_days" in result.columns:
        result["maintenance_days"] = result["maintenance_days"].clip(lower=0)
    for col in ["temperature", "pressure", "cycle_time", "vibration", "energy_consumption", "production_speed"]:
        if col in result.columns:
            result[col] = result[col].clip(lower=0.0)
    return result

def clean_dataframe(df, rename_columns=True, remove_duplicates=True, sort_chronological=True):
    result = df.copy()
    if rename_columns:
        result = clean_column_names(result)
    result = convert_numeric_like_columns(result)
    result = clean_categorical_values(result)
    result = validate_ranges(result)
    if remove_duplicates:
        result = result.drop_duplicates().reset_index(drop=True)
    for column in result.select_dtypes(include=np.number).columns:
        if result[column].isna().any():
            median = result[column].median()
            if pd.notna(median):
                result[column] = result[column].fillna(median)
    # Handle categorical / string columns missing values
    for column in result.select_dtypes(include=["object", "string"]).columns:
        col_lower = str(column).lower()
        if any(term in col_lower for term in ["time", "date"]):
            continue
        if result[column].isna().any():
            result[column] = result[column].fillna("Unknown")

    # Chronological sort and timestamp imputation
    if sort_chronological:
        has_batch = "batch_id" in result.columns and pd.api.types.is_numeric_dtype(result["batch_id"])
        ts_cols = [c for c in result.columns if any(t in str(c).lower() for t in ["timestamp", "datetime", "date_time"])]
        
        if ts_cols and has_batch:
            ts_col = ts_cols[0]
            result[ts_col] = pd.to_datetime(result[ts_col], errors="coerce")
            result = result.sort_values(by="batch_id").reset_index(drop=True)
            # If any timestamps were missing/unparseable, interpolate or reconstruct
            if result[ts_col].isna().any():
                result[ts_col] = result[ts_col].interpolate(method="linear")
        elif ts_cols:
            ts_col = ts_cols[0]
            result[ts_col] = pd.to_datetime(result[ts_col], errors="coerce")
            result = result.sort_values(by=ts_col).reset_index(drop=True)
        elif has_batch:
            result = result.sort_values(by="batch_id").reset_index(drop=True)
            
    return result
