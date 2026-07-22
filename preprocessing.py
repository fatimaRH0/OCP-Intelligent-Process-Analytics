"""
preprocessing.py
================
Automatic data ingestion, validation, cleaning and preparation.

The engineer only uploads a raw CSV/Excel file. This module is fully
responsible for:
    - Detecting the file format
    - Reading it safely
    - Auto-detecting the correct columns (via config.COLUMN_SYNONYMS)
    - Validating structure (required columns, minimum rows)
    - Converting dates
    - Handling missing values
    - Removing duplicates
    - Detecting and flagging abnormal (outlier) values
    - Producing a clean, ML-ready DataFrame

No manual intervention from the engineer is ever required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from io import BytesIO

import numpy as np
import pandas as pd

from config import (
    COLUMN_SYNONYMS, REQUIRED_COLUMNS, COL_DATE, COL_SO2, COL_DP, COL_TC,
    ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_ROWS_ALLOWED, MAX_COLUMNS_ALLOWED,
    THRESHOLDS,
)
from utils import AppError, safe_run, log_operation


# ======================================================================
# RESULT CONTAINER
# ======================================================================

@dataclass
class PreprocessingReport:
    original_rows: int
    original_columns: int
    final_rows: int
    final_columns: int
    detected_mapping: Dict[str, str]
    missing_values_filled: int
    duplicates_removed: int
    outliers_detected: int
    outlier_details: Dict[str, int] = field(default_factory=dict)
    date_column_found: bool = False
    warnings: List[str] = field(default_factory=list)


# ======================================================================
# 1. FILE LOADING
# ======================================================================

@safe_run("The uploaded file could not be read. Please verify it is a valid CSV or Excel export.")
def load_file(uploaded_file) -> pd.DataFrame:
    """
    Accepts a Streamlit UploadedFile (CSV, XLSX, or XLS), validates it,
    and returns a raw pandas DataFrame. Auto-detects format from the
    file extension — no manual configuration required.
    """
    filename = getattr(uploaded_file, "name", "uploaded_file")
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in ALLOWED_EXTENSIONS:
        raise AppError(
            f"Unsupported file type '{suffix}'. Please upload a CSV or Excel file "
            f"({', '.join(ALLOWED_EXTENSIONS)})."
        )

    # Basic size guard (Streamlit exposes .size on UploadedFile)
    size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise AppError(
            f"File is too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    raw_bytes = uploaded_file.read()
    buffer = BytesIO(raw_bytes)

    try:
        if suffix == ".csv":
            df = _read_csv_robust(buffer)
        else:
            df = pd.read_excel(buffer, engine=None)
    except Exception as exc:  # noqa: BLE001
        raise AppError(
            "The file appears to be corrupted or is not correctly formatted."
        ) from exc

    if df.empty:
        raise AppError("The uploaded file contains no data.")

    if df.shape[0] > MAX_ROWS_ALLOWED:
        raise AppError(f"File exceeds the maximum of {MAX_ROWS_ALLOWED:,} rows.")

    if df.shape[1] > MAX_COLUMNS_ALLOWED:
        raise AppError(f"File exceeds the maximum of {MAX_COLUMNS_ALLOWED} columns.")

    log_operation("file_uploaded", {"file_name": filename, "rows": df.shape[0], "cols": df.shape[1]})
    return df


def _read_csv_robust(buffer: BytesIO) -> pd.DataFrame:
    """Try common separators/encodings automatically."""
    for sep in (None, ";", ",", "\t"):
        buffer.seek(0)
        try:
            df = pd.read_csv(buffer, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    buffer.seek(0)
    return pd.read_csv(buffer)


# ======================================================================
# 2. COLUMN AUTO-DETECTION
# ======================================================================

def _normalize(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Map raw dataframe columns to canonical logical names (date, so2,
    delta_p, tc_global) using the synonym dictionary in config.py.
    Returns e.g. {"so2": "SO2_emission_mgNm3", "delta_p": "DP", ...}
    """
    normalized_lookup = {_normalize(c): c for c in df.columns}
    mapping: Dict[str, Optional[str]] = {}

    for logical_name, synonyms in COLUMN_SYNONYMS.items():
        found = None
        for syn in synonyms:
            syn_norm = _normalize(syn)
            for norm_col, original_col in normalized_lookup.items():
                if syn_norm == norm_col or syn_norm in norm_col or norm_col in syn_norm:
                    found = original_col
                    break
            if found:
                break
        mapping[logical_name] = found

    return mapping


# ======================================================================
# 3. VALIDATION
# ======================================================================

def validate_mapping(mapping: Dict[str, Optional[str]]) -> None:
    missing_logical = [
        logical for logical in ("so2", "delta_p", "tc_global")
        if mapping.get(logical) is None
    ]
    if missing_logical:
        pretty = ", ".join(m.upper() for m in missing_logical)
        raise AppError(
            f"The dataset is missing required process variable(s): {pretty}. "
            "Please verify the column names in your file (SO2, ΔP / Delta_P, TC_global)."
        )


# ======================================================================
# 4. OUTLIER DETECTION (IQR method — robust, no manual tuning needed)
# ======================================================================

def _detect_outliers_iqr(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


# ======================================================================
# 5. MAIN PIPELINE
# ======================================================================

@safe_run("An error occurred while preparing the dataset.")
def preprocess_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """
    Full automatic preprocessing pipeline. Returns a clean, canonical
    DataFrame with columns [date?, SO2, delta_P, TC_global] plus a
    PreprocessingReport describing every transformation applied.
    """
    original_rows, original_cols = raw_df.shape
    warnings: List[str] = []

    mapping = detect_columns(raw_df)
    validate_mapping(mapping)

    canonical_map = {}
    if mapping.get("date"):
        canonical_map[mapping["date"]] = COL_DATE
    canonical_map[mapping["so2"]] = COL_SO2
    canonical_map[mapping["delta_p"]] = COL_DP
    canonical_map[mapping["tc_global"]] = COL_TC

    df = raw_df.rename(columns=canonical_map)
    keep_cols = [c for c in [COL_DATE, COL_SO2, COL_DP, COL_TC] if c in df.columns]
    df = df[keep_cols].copy()

    # ---- Date handling ----
    date_found = COL_DATE in df.columns
    if date_found:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
        n_bad_dates = df[COL_DATE].isna().sum()
        if n_bad_dates > 0:
            warnings.append(f"{n_bad_dates} row(s) had an unreadable date and were timestamped as unknown.")
    else:
        warnings.append("No date/timestamp column detected — chronological trend analysis will use row order.")

    # ---- Numeric coercion ----
    for col in (COL_SO2, COL_DP, COL_TC):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Missing values ----
    missing_before = int(df[[COL_SO2, COL_DP, COL_TC]].isna().sum().sum())
    for col in (COL_SO2, COL_DP, COL_TC):
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # ---- Duplicates ----
    rows_before = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = rows_before - df.shape[0]

    # ---- Drop rows still fully invalid (should be rare after fill) ----
    df = df.dropna(subset=[COL_SO2, COL_DP, COL_TC], how="all")

    # ---- Outlier detection (flagged, not removed — engineer needs to see them) ----
    outlier_details: Dict[str, int] = {}
    total_outliers = 0
    for col in (COL_SO2, COL_DP, COL_TC):
        mask = _detect_outliers_iqr(df[col])
        outlier_details[col] = int(mask.sum())
        total_outliers += int(mask.sum())
        df[f"{col}_is_outlier"] = mask

    if total_outliers > 0:
        warnings.append(
            f"{total_outliers} statistically abnormal value(s) were detected across process variables "
            "and flagged for engineering review (not removed)."
        )

    # ---- Sort chronologically if possible ----
    if date_found:
        df = df.sort_values(COL_DATE).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if df.shape[0] < 5:
        raise AppError("After cleaning, too few valid observations remain to perform a reliable analysis.")

    report = PreprocessingReport(
        original_rows=original_rows,
        original_columns=original_cols,
        final_rows=df.shape[0],
        final_columns=df.shape[1],
        detected_mapping={k: v for k, v in mapping.items() if v},
        missing_values_filled=missing_before,
        duplicates_removed=duplicates_removed,
        outliers_detected=total_outliers,
        outlier_details=outlier_details,
        date_column_found=date_found,
        warnings=warnings,
    )

    log_operation("dataset_prepared", {
        "rows_in": original_rows, "rows_out": df.shape[0],
        "duplicates_removed": duplicates_removed, "outliers": total_outliers,
    })

    return df, report
