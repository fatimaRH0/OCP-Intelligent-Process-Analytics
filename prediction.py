"""
prediction.py
=============
Machine Learning module: automatically trains a Multiple Linear
Regression model predicting SO2 from Delta_P and TC_global, evaluates
it, generates predictions for every observation, and explains the
result in plain industrial language (never exposing ML jargon to the
engineer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

from config import COL_SO2, COL_DP, COL_TC, ML_CONFIG, MODEL_QUALITY_BANDS, MODEL_ARTIFACT_PATH
from utils import AppError, safe_run, log_operation


@dataclass
class RegressionResult:
    intercept: float
    coef_dp: float
    coef_tc: float
    equation_text: str
    r2: float
    rmse: float
    mae: float
    quality_label: str
    predictions: pd.Series
    residuals: pd.Series
    accuracy_pct: float
    n_train: int
    n_test: int
    interpretation: str


def _quality_label(r2: float) -> str:
    for threshold, label in MODEL_QUALITY_BANDS:
        if r2 >= threshold:
            return label
    return "Weak"


def _build_interpretation(coef_dp: float, coef_tc: float, r2: float, quality: str, accuracy_pct: float) -> str:
    dp_effect = "increases" if coef_dp > 0 else "decreases"
    tc_effect = "increases" if coef_tc > 0 else "decreases"

    dp_strength = "strong" if abs(coef_dp) > 0.5 else ("moderate" if abs(coef_dp) > 0.1 else "weak")
    tc_strength = "strong" if abs(coef_tc) > 0.5 else ("moderate" if abs(coef_tc) > 0.1 else "weak")

    text = (
        f"The predictive model explains {r2*100:.1f}% of the variation in SO₂ emissions using "
        f"Pressure Drop and Global Conversion Rate as inputs — this is considered '{quality.lower()}' "
        f"predictive performance, with an approximate prediction accuracy of {accuracy_pct:.1f}%.\n\n"
        f"When Pressure Drop rises, SO₂ tends to {dp_effect} (a {dp_strength} effect). "
        f"When the Global Conversion Rate rises, SO₂ tends to {tc_effect} (a {tc_strength} effect). "
    )

    if quality in ("Excellent", "Good"):
        text += (
            "This means the model can be reasonably trusted to anticipate SO₂ behavior from "
            "process conditions, supporting proactive operating decisions."
        )
    elif quality == "Acceptable":
        text += (
            "The model gives a useful directional signal but should be combined with engineering "
            "judgment, as a meaningful share of SO₂ variation is not explained by these two variables alone."
        )
    else:
        text += (
            "The model's explanatory power is limited, suggesting SO₂ emissions are driven "
            "significantly by factors beyond Pressure Drop and Conversion Rate (e.g. catalyst "
            "condition, feed gas composition, or furnace performance). Predictions should be "
            "treated as indicative only."
        )

    return text


@safe_run("An error occurred while training the predictive model.")
def train_regression_model(df: pd.DataFrame) -> RegressionResult:
    """
    Train a Multiple Linear Regression model: SO2 ~ Delta_P + TC_global.
    Automatically splits data, fits, evaluates, predicts for every
    observation, and produces a plain-language interpretation.
    """
    data = df[[COL_SO2, COL_DP, COL_TC]].apply(pd.to_numeric, errors="coerce").dropna()

    if data.shape[0] < ML_CONFIG["min_rows_required"]:
        raise AppError(
            f"At least {ML_CONFIG['min_rows_required']} valid observations are required to train "
            f"a reliable predictive model; only {data.shape[0]} are available in this dataset."
        )

    X = data[[COL_DP, COL_TC]].values
    y = data[COL_SO2].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ML_CONFIG["test_size"], random_state=ML_CONFIG["random_state"]
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_test_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    mae = float(mean_absolute_error(y_test, y_test_pred))

    # Predict on the full dataset for the "Real vs Predicted" and residual views
    full_predictions = model.predict(X)
    residuals = data[COL_SO2].values - full_predictions

    mean_so2 = float(data[COL_SO2].mean()) or 1.0
    accuracy_pct = max(0.0, min(100.0, 100.0 * (1 - (mae / mean_so2))))

    intercept = float(model.intercept_)
    coef_dp, coef_tc = float(model.coef_[0]), float(model.coef_[1])

    equation = (
        f"SO₂ = {intercept:.3f} "
        f"{'+' if coef_dp >= 0 else '-'} {abs(coef_dp):.4f} × ΔP "
        f"{'+' if coef_tc >= 0 else '-'} {abs(coef_tc):.4f} × TC_global"
    )

    quality = _quality_label(r2)
    interpretation = _build_interpretation(coef_dp, coef_tc, r2, quality, accuracy_pct)

    # Persist the trained model artifact
    try:
        joblib.dump(model, MODEL_ARTIFACT_PATH)
    except OSError:
        pass  # non-critical; app can still run without persisted artifact

    log_operation("model_trained", {"r2": round(r2, 4), "rmse": round(rmse, 2), "n_rows": int(data.shape[0])})

    return RegressionResult(
        intercept=intercept,
        coef_dp=coef_dp,
        coef_tc=coef_tc,
        equation_text=equation,
        r2=r2,
        rmse=rmse,
        mae=mae,
        quality_label=quality,
        predictions=pd.Series(full_predictions, index=data.index, name="SO2_Predicted"),
        residuals=pd.Series(residuals, index=data.index, name="Residual"),
        accuracy_pct=accuracy_pct,
        n_train=len(X_train),
        n_test=len(X_test),
        interpretation=interpretation,
    )


def build_prediction_table(df: pd.DataFrame, result: RegressionResult) -> pd.DataFrame:
    """
    Build the full Real vs Predicted table, aligned back to the
    original dataframe rows, for export (CSV/Excel) and display.
    """
    out = df.loc[result.predictions.index].copy()
    out["SO2_Real"] = out[COL_SO2]
    out["SO2_Predicted"] = result.predictions.round(2)
    out["Residual"] = result.residuals.round(2)
    out["Absolute_Error"] = out["Residual"].abs().round(2)
    cols_to_show = [c for c in out.columns if c not in (COL_SO2,)]
    return out[cols_to_show]


def forecast_next_risk(result: RegressionResult, recent_trend_slope: float) -> str:
    """
    Simple forward-looking statement combining model residual trend
    with recent SO2 slope, used by the alert system for a
    'predictive' style warning (without exposing time-series jargon).
    """
    if recent_trend_slope > 0 and result.r2 >= 0.5:
        return (
            "Based on the recent trend and the trained model, SO₂ emissions are projected to "
            "continue rising if current operating conditions persist."
        )
    elif recent_trend_slope > 0:
        return (
            "SO₂ emissions show a rising trend, though the current model has limited confidence "
            "in projecting this forward."
        )
    else:
        return "SO₂ emissions show no upward trend based on recent data; no rising-risk forecast applies."
