"""
visualization.py
================
All Plotly visualizations for the platform, each paired with an
automatically generated plain-language explanation. The engineer
never has to interpret a graph manually — every chart-producing
function returns both the Plotly figure and a caption-ready
interpretation string.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as scipy_stats

from config import Colors, COL_SO2, COL_DP, COL_TC, COL_DATE, THRESHOLDS

PLOT_TEMPLATE = "plotly_white"
FONT = dict(family="Segoe UI, Roboto, Arial, sans-serif", size=13, color=Colors.TEXT_DARK)


def _base_layout(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=Colors.DARK_BLUE, family=FONT["family"]), x=0.02),
        template=PLOT_TEMPLATE,
        font=FONT,
        height=height,
        margin=dict(l=40, r=30, t=60, b=40),
        plot_bgcolor="rgba(255,255,255,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    return fig


# ======================================================================
# 1. TIME EVOLUTION
# ======================================================================

def chart_evolution(df: pd.DataFrame, column: str, label: str) -> Tuple[go.Figure, str]:
    x = df[COL_DATE] if COL_DATE in df.columns else df.index
    y = df[column]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=label,
        line=dict(color=Colors.DARK_BLUE, width=2),
        marker=dict(size=4, color=Colors.OCP_GREEN),
    ))
    th = THRESHOLDS.get(column)
    if th:
        fig.add_hline(y=th.normal_max, line_dash="dot", line_color=Colors.STATUS_ORANGE,
                       annotation_text="Normal limit", annotation_position="top left")
    fig = _base_layout(fig, f"{label} — Evolution Over Time")
    fig.update_xaxes(title="Observation / Date")
    fig.update_yaxes(title=f"{label} ({th.unit if th else ''})")

    slope = _trend_slope(y.values)
    trend_word = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")
    explanation = (
        f"{label} shows a {trend_word} trend across the analyzed period. "
        f"{'This upward drift should be monitored closely.' if slope > 0 and th and th.lower_is_better else ''}"
    ).strip()
    return fig, explanation


def _trend_slope(values: np.ndarray) -> float:
    x = np.arange(len(values))
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return 0.0
    slope, _intercept, _r, _p, _se = scipy_stats.linregress(x[valid], values[valid])
    return float(slope)


# ======================================================================
# 2. SCATTER PLOTS (feature vs SO2)
# ======================================================================

def chart_scatter(df: pd.DataFrame, x_col: str, x_label: str) -> Tuple[go.Figure, str]:
    fig = px.scatter(
        df, x=x_col, y=COL_SO2, trendline="ols",
        color_discrete_sequence=[Colors.OCP_GREEN],
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color=Colors.DARK_BLUE)))
    for trace in fig.data:
        if trace.mode == "lines":
            trace.line.color = Colors.DARK_BLUE
            trace.line.width = 3
    fig = _base_layout(fig, f"SO₂ vs {x_label}")
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="SO₂ Emission")

    corr = df[[x_col, COL_SO2]].corr().iloc[0, 1]
    strength = "strong" if abs(corr) > 0.6 else ("moderate" if abs(corr) > 0.3 else "weak")
    direction = "positive" if corr > 0 else "negative"
    explanation = (
        f"The scatter plot shows a {strength} {direction} relationship between {x_label} and SO₂ "
        f"(r = {corr:.2f}). {'This variable meaningfully influences emissions.' if strength != 'weak' else 'This variable has limited direct influence on emissions.'}"
    )
    return fig, explanation


# ======================================================================
# 3. HISTOGRAM
# ======================================================================

def chart_histogram(df: pd.DataFrame, column: str, label: str) -> Tuple[go.Figure, str]:
    fig = px.histogram(df, x=column, nbins=30, color_discrete_sequence=[Colors.DARK_BLUE])
    fig = _base_layout(fig, f"{label} — Distribution")
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="Frequency")

    skew = float(df[column].skew())
    shape = "roughly symmetric" if abs(skew) < 0.5 else ("right-skewed (occasional high spikes)" if skew > 0 else "left-skewed (occasional low dips)")
    explanation = f"The distribution of {label} is {shape}, based on the observed data."
    return fig, explanation


# ======================================================================
# 4. BOXPLOT
# ======================================================================

def chart_boxplot(df: pd.DataFrame, column: str, label: str) -> Tuple[go.Figure, str]:
    fig = px.box(df, y=column, points="outliers", color_discrete_sequence=[Colors.OCP_GREEN])
    fig = _base_layout(fig, f"{label} — Spread & Outliers", height=380)
    fig.update_yaxes(title=label)

    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    n_outliers = int(((df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)).sum())
    explanation = (
        f"The boxplot shows {n_outliers} statistically abnormal observation(s) for {label}. "
        f"{'These should be reviewed for measurement errors or genuine process excursions.' if n_outliers > 0 else 'No abnormal values were detected.'}"
    )
    return fig, explanation


# ======================================================================
# 5. HEATMAP (correlation matrix)
# ======================================================================

def chart_heatmap(corr_matrix: pd.DataFrame) -> Tuple[go.Figure, str]:
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=Colors.HEATMAP_SCALE,
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    fig = _base_layout(fig, "Correlation Heatmap — Process Variables", height=420)

    strongest_pair = _strongest_offdiag(corr_matrix)
    explanation = (
        f"The heatmap summarizes how strongly each process variable relates to the others. "
        f"The strongest relationship is between {strongest_pair[0]} and {strongest_pair[1]} "
        f"(r = {strongest_pair[2]:.2f})."
    )
    return fig, explanation


def _strongest_offdiag(corr: pd.DataFrame):
    best = (None, None, 0.0)
    for i in corr.index:
        for j in corr.columns:
            if i != j and abs(corr.loc[i, j]) > abs(best[2]):
                best = (i, j, corr.loc[i, j])
    return best


# ======================================================================
# 6. REAL vs PREDICTED
# ======================================================================

def chart_real_vs_predicted(y_real: pd.Series, y_pred: pd.Series, r2: float) -> Tuple[go.Figure, str]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_real, y=y_pred, mode="markers", name="Observations",
        marker=dict(size=7, color=Colors.OCP_GREEN, opacity=0.75, line=dict(width=0.5, color=Colors.DARK_BLUE)),
    ))
    min_v, max_v = float(min(y_real.min(), y_pred.min())), float(max(y_real.max(), y_pred.max()))
    fig.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect Prediction",
        line=dict(color=Colors.STATUS_RED, dash="dash", width=2),
    ))
    fig = _base_layout(fig, "Real vs Predicted SO₂")
    fig.update_xaxes(title="Real SO₂")
    fig.update_yaxes(title="Predicted SO₂")

    quality = "closely" if r2 >= 0.7 else ("reasonably" if r2 >= 0.5 else "loosely")
    explanation = (
        f"Predicted values track {quality} with real SO₂ measurements (R² = {r2:.2f}). "
        f"Points closer to the dashed line indicate more accurate predictions."
    )
    return fig, explanation


# ======================================================================
# 7. RESIDUAL PLOT
# ======================================================================

def chart_residuals(y_pred: pd.Series, residuals: pd.Series) -> Tuple[go.Figure, str]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(size=7, color=Colors.DARK_BLUE, opacity=0.7),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=Colors.STATUS_RED)
    fig = _base_layout(fig, "Residual Plot — Prediction Errors")
    fig.update_xaxes(title="Predicted SO₂")
    fig.update_yaxes(title="Residual (Real − Predicted)")

    mean_resid = float(residuals.mean())
    pattern = "randomly scattered around zero, indicating a well-behaved model" if abs(mean_resid) < residuals.std() * 0.2 else "showing a slight systematic bias"
    explanation = f"Prediction errors are {pattern}. No major structural bias is expected in typical operating ranges." if "random" in pattern else f"Prediction errors are {pattern}, meaning the model may consistently over- or under-estimate SO₂ in certain ranges."
    return fig, explanation


# ======================================================================
# 8. TREND ANALYSIS (multi-variable overlay, normalized)
# ======================================================================

def chart_trend_overlay(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    x = df[COL_DATE] if COL_DATE in df.columns else df.index
    fig = go.Figure()
    colors = [Colors.DARK_BLUE, Colors.OCP_GREEN, Colors.STATUS_ORANGE]
    for col, color in zip((COL_SO2, COL_DP, COL_TC), colors):
        series = df[col]
        normalized = (series - series.min()) / (series.max() - series.min() + 1e-9)
        fig.add_trace(go.Scatter(x=x, y=normalized, mode="lines", name=col, line=dict(color=color, width=2)))
    fig = _base_layout(fig, "Normalized Trend Comparison (SO₂, ΔP, TC)", height=420)
    fig.update_yaxes(title="Normalized scale (0–1)")

    explanation = (
        "This chart overlays all three process variables on a common scale to reveal whether "
        "they move together or diverge over time — helping identify shared root causes of change."
    )
    return fig, explanation
