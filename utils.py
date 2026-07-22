"""
utils.py
========
Shared utility functions used across the OCP Intelligent Process Analytics
platform: safe error handling, operation logging, formatting helpers,
and generic validation utilities.

Keeping these in one place avoids duplicated code across
preprocessing.py, prediction.py, report.py, dashboard.py, etc.
"""

from __future__ import annotations

import json
import traceback
import functools
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from config import OPERATION_LOG_DB, HISTORY_DB


# ======================================================================
# 1. SAFE EXECUTION WRAPPER
# ======================================================================
# The engineer must never see a raw Python traceback. Every risky
# operation in the app should be wrapped with this decorator so that
# internal errors are logged (not exposed) and a clean message is
# returned instead.

class AppError(Exception):
    """A controlled, user-facing application error."""
    def __init__(self, message: str, technical_detail: str = ""):
        super().__init__(message)
        self.message = message
        self.technical_detail = technical_detail


def safe_run(user_facing_message: str = "An unexpected error occurred while processing the data."):
    """
    Decorator that catches any exception, logs the technical traceback
    internally, and raises a clean AppError with a safe, non-technical
    message for the UI layer to display.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AppError:
                raise
            except Exception as exc:  # noqa: BLE001
                _log_internal_error(func.__name__, exc)
                raise AppError(user_facing_message, technical_detail=str(exc)) from exc
        return wrapper
    return decorator


def _log_internal_error(function_name: str, exc: Exception) -> None:
    """Write full technical error details to a local log file only —
    never shown to the engineer."""
    error_log_path = OPERATION_LOG_DB.parent / "error_log.txt"
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().isoformat()}] Error in {function_name}\n")
        f.write(traceback.format_exc())
        f.write("\n" + "-" * 80 + "\n")


# ======================================================================
# 2. OPERATION LOG (audit trail of user actions)
# ======================================================================

def log_operation(action: str, details: Optional[dict] = None) -> None:
    """
    Append an entry to the operation log: uploads, analyses, predictions,
    report generation, exports, etc. Used by the 'Operation Log' feature.
    """
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details or {},
    }
    log = _read_json_list(OPERATION_LOG_DB)
    log.append(entry)
    _write_json_list(OPERATION_LOG_DB, log)


def get_operation_log() -> pd.DataFrame:
    """Return the operation log as a DataFrame, most recent first."""
    log = _read_json_list(OPERATION_LOG_DB)
    if not log:
        return pd.DataFrame(columns=["timestamp", "action", "details"])
    df = pd.DataFrame(log)
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


# ======================================================================
# 3. ANALYSIS HISTORY (store & recall past analyses)
# ======================================================================

def save_analysis_to_history(record: dict) -> None:
    """
    Persist a summary of a completed analysis so the engineer can
    reopen and compare it later. `record` should contain at minimum:
    date, time, file_name, n_observations, model_r2, risk_level.
    """
    record = dict(record)
    record.setdefault("id", datetime.now().strftime("%Y%m%d%H%M%S%f"))
    record.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
    record.setdefault("time", datetime.now().strftime("%H:%M:%S"))

    history = _read_json_list(HISTORY_DB)
    history.append(record)
    _write_json_list(HISTORY_DB, history)
    log_operation("analysis_saved", {"id": record["id"], "file_name": record.get("file_name")})


def load_analysis_history() -> pd.DataFrame:
    """Return all stored analyses as a DataFrame, most recent first."""
    history = _read_json_list(HISTORY_DB)
    if not history:
        return pd.DataFrame(columns=["id", "date", "time", "file_name",
                                      "n_observations", "model_r2", "risk_level"])
    df = pd.DataFrame(history)
    sort_col = "date" if "date" in df.columns else df.columns[0]
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


def get_analysis_by_id(analysis_id: str) -> Optional[dict]:
    history = _read_json_list(HISTORY_DB)
    for record in history:
        if record.get("id") == analysis_id:
            return record
    return None


# ---- internal JSON helpers -------------------------------------------------

def _read_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_json_list(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


# ======================================================================
# 4. FORMATTING HELPERS
# ======================================================================

def fmt_number(value: Any, decimals: int = 2) -> str:
    """Format a number for display, gracefully handling None/NaN."""
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_percent(value: Any, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_timestamp(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def human_bytes(num_bytes: int) -> str:
    """Convert a byte count into a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ======================================================================
# 5. GENERIC VALIDATION HELPERS
# ======================================================================

def is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises ZeroDivisionError."""
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
