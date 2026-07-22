"""
statistics.py
=============
Automatic descriptive statistical analysis of the process variables
(SO2, Delta_P, TC_global), paired with plain-language industrial
interpretation for every metric. The engineer never has to interpret
a raw number themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import numpy as np

from config import COL_SO2, COL_DP, COL_TC, THRESHOLDS
from utils import safe_run


@dataclass
class VariableStats:
    column: str
    label: str
    unit: str
    mean: float
    median: float
    std: float
    variance: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    status: str          # "Normal" | "Warning" | "Critical"
    interpretation: str  # plain-language explanation


@dataclass
class StatisticsReport:
    variables: Dict[str, VariableStats] = field(default_factory=dict)
    n_observations: int = 0


def _classify_and_explain(column: str, mean_value: float) -> tuple[str, str]:
    """
    Classify the mean of a variable against industrial thresholds and
    generate a plain-language explanation, exactly as a process
    engineer would phrase it.
    """
    th = THRESHOLDS[column]

    if th.lower_is_better:
        if mean_value <= th.normal_max:
            status = "Normal"
            explanation = (
                f"The average {th.name} ({mean_value:.1f} {th.unit}) is within the normal "
                f"operating range (≤ {th.normal_max} {th.unit}). No corrective action required."
            )
        elif mean_value <= th.warning_max:
            status = "Warning"
            explanation = (
                f"The average {th.name} ({mean_value:.1f} {th.unit}) is elevated, between "
                f"{th.normal_max} and {th.warning_max} {th.unit}. This should be monitored closely "
                "as it may indicate early process drift."
            )
        else:
            status = "Critical"
            explanation = (
                f"The average {th.name} ({mean_value:.1f} {th.unit}) exceeds the warning limit of "
                f"{th.warning_max} {th.unit}. This is high and requires engineering attention."
            )
    else:
        # Higher is better (e.g. conversion rate)
        if mean_value >= th.normal_max:
            status = "Normal"
            explanation = (
                f"The average {th.name} ({mean_value:.2f} {th.unit}) is healthy, at or above the "
                f"target of {th.normal_max}{th.unit}. Conversion performance is satisfactory."
            )
        elif mean_value >= th.warning_max:
            status = "Warning"
            explanation = (
                f"The average {th.name} ({mean_value:.2f} {th.unit}) has slipped slightly below "
                f"target ({th.normal_max}{th.unit}), currently at {mean_value:.2f}{th.unit}. "
                "This should be watched — it may reflect catalyst aging or gas flow imbalance."
            )
        else:
            status = "Critical"
            explanation = (
                f"The average {th.name} ({mean_value:.2f} {th.unit}) is below the acceptable "
                f"threshold of {th.warning_max}{th.unit}. This is a significant conversion loss "
                "and should be investigated promptly."
            )

    return status, explanation


@safe_run("An error occurred while computing statistical indicators.")
def compute_statistics(df: pd.DataFrame) -> StatisticsReport:
    """
    Compute Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3 for the
    three core process variables, and attach an automatic industrial
    interpretation to each.
    """
    report = StatisticsReport(n_observations=int(df.shape[0]))

    labels = {
        COL_SO2: ("SO₂ Emission", THRESHOLDS[COL_SO2].unit),
        COL_DP: ("Pressure Drop (ΔP)", THRESHOLDS[COL_DP].unit),
        COL_TC: ("Global Conversion Rate (TC)", THRESHOLDS[COL_TC].unit),
    }

    for col in (COL_SO2, COL_DP, COL_TC):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        mean_v = float(series.mean())
        status, explanation = _classify_and_explain(col, mean_v)

        label, unit = labels[col]
        report.variables[col] = VariableStats(
            column=col,
            label=label,
            unit=unit,
            mean=mean_v,
            median=float(series.median()),
            std=float(series.std()),
            variance=float(series.var()),
            minimum=float(series.min()),
            maximum=float(series.max()),
            q1=float(series.quantile(0.25)),
            q3=float(series.quantile(0.75)),
            status=status,
            interpretation=explanation,
        )

    return report


def overall_variability_comment(report: StatisticsReport) -> str:
    """
    A single synthesized sentence about process stability, based on
    the coefficient of variation of SO2 (the primary environmental KPI).
    """
    so2 = report.variables.get(COL_SO2)
    if so2 is None or so2.mean == 0:
        return "Insufficient data to assess process variability."

    cv = so2.std / so2.mean  # coefficient of variation
    if cv < 0.10:
        return "SO₂ emissions are very stable over the analyzed period, indicating consistent process control."
    elif cv < 0.25:
        return "SO₂ emissions show moderate variability, suggesting some fluctuation in operating conditions."
    else:
        return (
            "SO₂ emissions are highly variable over the analyzed period. This instability suggests "
            "inconsistent process control and warrants investigation of operating practices."
        )
