"""
recommendation.py
==================
The intelligence core of the platform:
    - Intelligent Alert System (Green / Orange / Red)
    - Industrial Performance Score (0-100)
    - AI Decision Assistant (production status + root cause narrative)
    - Predictive Maintenance suggestions
    - Automatic engineering recommendations

Every function here translates numbers into decisions and actions,
exactly as an experienced process engineer would reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import pandas as pd

from config import (
    COL_SO2, COL_DP, COL_TC, THRESHOLDS, PERFORMANCE_SCORE_WEIGHTS,
    SCORE_BANDS, MAINTENANCE_RULES, Colors,
)
from statistics import StatisticsReport
from correlation import CorrelationReport, dominant_driver_of_so2
from prediction import RegressionResult


# ======================================================================
# DATA CONTAINERS
# ======================================================================

@dataclass
class Alert:
    level: str        # "Normal" | "Warning" | "Critical"
    color: str
    icon: str
    title: str
    message: str
    recommendation: str


@dataclass
class PerformanceScore:
    score: float
    band_label: str
    band_color: str
    breakdown: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class DecisionAssistantSummary:
    status: str        # "Stable" | "Attention Needed" | "Critical"
    status_color: str
    reasoning: str
    dominant_driver: str
    equipment_to_inspect: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


# ======================================================================
# 1. INTELLIGENT ALERT SYSTEM
# ======================================================================

def generate_alerts(stats: StatisticsReport, forecast_note: str = "") -> List[Alert]:
    alerts: List[Alert] = []

    level_map = {
        "Normal": (Colors.STATUS_GREEN, "🟢"),
        "Warning": (Colors.STATUS_ORANGE, "🟠"),
        "Critical": (Colors.STATUS_RED, "🔴"),
    }

    for col, var in stats.variables.items():
        color, icon = level_map[var.status]
        if var.status == "Normal":
            recommendation = "Continue standard monitoring. No intervention required."
        elif var.status == "Warning":
            recommendation = _warning_recommendation(col)
        else:
            recommendation = _critical_recommendation(col)

        alerts.append(Alert(
            level=var.status, color=color, icon=icon,
            title=f"{var.label}: {var.status}",
            message=var.interpretation,
            recommendation=recommendation,
        ))

    if forecast_note and "projected to continue rising" in forecast_note.lower():
        alerts.append(Alert(
            level="Warning", color=Colors.STATUS_ORANGE, icon="🟠",
            title="Predictive Risk Signal",
            message=forecast_note,
            recommendation="Proactively review operating conditions before the trend worsens.",
        ))

    severity_order = {"Critical": 0, "Warning": 1, "Normal": 2}
    alerts.sort(key=lambda a: severity_order[a.level])
    return alerts


def _warning_recommendation(column: str) -> str:
    mapping = {
        COL_SO2: "Monitor catalyst performance and gas flow; consider a preventive inspection.",
        COL_DP: "Schedule an inspection of filters and catalyst bed for early-stage clogging.",
        COL_TC: "Check gas flow distribution and furnace combustion efficiency.",
    }
    return mapping.get(column, "Increase monitoring frequency.")


def _critical_recommendation(column: str) -> str:
    mapping = {
        COL_SO2: "Inspect catalyst activity and converter beds immediately; verify absorption tower efficiency.",
        COL_DP: "Inspect gas blower and ductwork for fouling, blockage, or mechanical wear without delay.",
        COL_TC: "Inspect catalyst bed temperature profile and heat exchangers; conversion loss requires prompt action.",
    }
    return mapping.get(column, "Immediate engineering review required.")


# ======================================================================
# 2. INDUSTRIAL PERFORMANCE SCORE (0-100)
# ======================================================================

def _component_score(column: str, mean_value: float) -> float:
    """Score a single variable from 0 (worst) to 100 (best) based on thresholds."""
    th = THRESHOLDS[column]
    if th.lower_is_better:
        if mean_value <= th.normal_max:
            return 100.0
        if mean_value >= th.warning_max * 1.5:
            return 0.0
        # linear decay between normal_max and 1.5x warning_max
        span = (th.warning_max * 1.5) - th.normal_max
        return max(0.0, 100.0 * (1 - (mean_value - th.normal_max) / span))
    else:
        if mean_value >= th.normal_max:
            return 100.0
        floor = th.warning_max * 0.98
        if mean_value <= floor:
            return 0.0
        span = th.normal_max - floor
        return max(0.0, 100.0 * (mean_value - floor) / span)


def compute_performance_score(stats: StatisticsReport, model: RegressionResult) -> PerformanceScore:
    so2_score = _component_score(COL_SO2, stats.variables[COL_SO2].mean)
    dp_score = _component_score(COL_DP, stats.variables[COL_DP].mean)
    tc_score = _component_score(COL_TC, stats.variables[COL_TC].mean)
    model_score = max(0.0, min(100.0, model.r2 * 100))

    weights = PERFORMANCE_SCORE_WEIGHTS
    total = (
        so2_score * weights["so2"]
        + dp_score * weights["delta_p"]
        + tc_score * weights["tc_global"]
        + model_score * weights["model_confidence"]
    )
    total = round(total, 1)

    band_label, band_color = "Poor", Colors.STATUS_RED
    for threshold, label, color in SCORE_BANDS:
        if total >= threshold:
            band_label, band_color = label, color
            break

    explanation = (
        f"The Industrial Performance Score ({total}/100 — {band_label}) combines SO₂ emission level "
        f"({weights['so2']*100:.0f}% weight), Pressure Drop ({weights['delta_p']*100:.0f}%), "
        f"Conversion Rate ({weights['tc_global']*100:.0f}%), and predictive model confidence "
        f"({weights['model_confidence']*100:.0f}%) into a single health indicator for Ligne H."
    )

    return PerformanceScore(
        score=total, band_label=band_label, band_color=band_color,
        breakdown={"SO2": so2_score, "Delta_P": dp_score, "TC_global": tc_score, "Model": model_score},
        explanation=explanation,
    )


# ======================================================================
# 3. AI DECISION ASSISTANT
# ======================================================================

def build_decision_assistant(
    stats: StatisticsReport,
    correlation: CorrelationReport,
    model: RegressionResult,
    score: PerformanceScore,
) -> DecisionAssistantSummary:

    critical_vars = [v for v in stats.variables.values() if v.status == "Critical"]
    warning_vars = [v for v in stats.variables.values() if v.status == "Warning"]

    if critical_vars:
        status, status_color = "Critical", Colors.STATUS_RED
    elif warning_vars:
        status, status_color = "Attention Needed", Colors.STATUS_ORANGE
    else:
        status, status_color = "Stable", Colors.STATUS_GREEN

    dominant = dominant_driver_of_so2(correlation)

    if status == "Critical":
        causes = ", ".join(v.label for v in critical_vars)
        reasoning = (
            f"Production is classified as CRITICAL because the following variable(s) exceed safe "
            f"operating limits: {causes}. The primary statistical driver of SO₂ behavior in this "
            f"dataset is {dominant}. Immediate engineering intervention is recommended."
        )
    elif status == "Attention Needed":
        causes = ", ".join(v.label for v in warning_vars)
        reasoning = (
            f"Production is STABLE overall but requires attention: {causes} are trending outside "
            f"the optimal range. The primary statistical driver of SO₂ behavior is {dominant}. "
            "Preventive action now can avoid escalation to a critical state."
        )
    else:
        reasoning = (
            f"All monitored process variables are within their normal operating ranges. "
            f"The primary statistical driver of SO₂ behavior is {dominant}. Production on Ligne H "
            "is considered stable; routine monitoring is sufficient."
        )

    equipment = _equipment_to_inspect(stats)
    actions = _priority_actions(stats, score)

    return DecisionAssistantSummary(
        status=status, status_color=status_color, reasoning=reasoning,
        dominant_driver=dominant, equipment_to_inspect=equipment, actions=actions,
    )


def _equipment_to_inspect(stats: StatisticsReport) -> List[str]:
    equipment = []
    if stats.variables[COL_SO2].status != "Normal":
        equipment += ["Catalyst beds (converter)", "Absorption tower"]
    if stats.variables[COL_DP].status != "Normal":
        equipment += ["Gas blower", "Filters / ductwork"]
    if stats.variables[COL_TC].status != "Normal":
        equipment += ["Heat exchangers", "Furnace / combustion chamber"]
    return sorted(set(equipment)) or ["No specific equipment flagged — routine inspection schedule applies."]


def _priority_actions(stats: StatisticsReport, score: PerformanceScore) -> List[str]:
    actions = []
    for col, var in stats.variables.items():
        if var.status == "Critical":
            actions.append(_critical_recommendation(col))
        elif var.status == "Warning":
            actions.append(_warning_recommendation(col))
    if score.score < 50:
        actions.append("Escalate to shift supervisor: overall plant health score is in the 'Poor' range.")
    if not actions:
        actions.append("Maintain current operating parameters and continue standard monitoring routine.")
    return actions


# ======================================================================
# 4. PREDICTIVE MAINTENANCE SUGGESTIONS
# ======================================================================

def predictive_maintenance_suggestions(stats: StatisticsReport, model: RegressionResult) -> List[dict]:
    suggestions: List[dict] = []
    conditions_active = set()

    if stats.variables[COL_SO2].status == "Critical":
        conditions_active.add("so2_critical")
    if stats.variables[COL_DP].status == "Critical":
        conditions_active.add("dp_critical")
    elif stats.variables[COL_DP].status == "Warning":
        conditions_active.add("dp_warning")
    if stats.variables[COL_TC].status == "Critical":
        conditions_active.add("tc_critical")
    elif stats.variables[COL_TC].status == "Warning":
        conditions_active.add("tc_warning")
    if model.r2 < 0.5:
        conditions_active.add("model_low_confidence")

    for rule in MAINTENANCE_RULES:
        if rule["condition"] in conditions_active:
            suggestions.append(rule)

    if not suggestions:
        suggestions.append({
            "condition": "nominal",
            "action": "No preventive maintenance action required at this time; continue routine inspection schedule.",
            "priority": "Low",
        })

    return suggestions


# ======================================================================
# 5. EXECUTIVE-LEVEL RECOMMENDATION LIST (used in dashboard + report)
# ======================================================================

def generate_recommendations(
    stats: StatisticsReport, correlation: CorrelationReport, model: RegressionResult, score: PerformanceScore
) -> List[str]:
    """Flat, deduplicated list of actionable recommendations for display."""
    recs: List[str] = []
    for col, var in stats.variables.items():
        if var.status == "Critical":
            recs.append(_critical_recommendation(col))
        elif var.status == "Warning":
            recs.append(_warning_recommendation(col))

    strongest = correlation.pairs[0] if correlation.pairs else None
    if strongest and strongest.strength in ("Strong", "Very strong"):
        recs.append(
            f"Prioritize control of {strongest.label_a if strongest.label_a != 'SO₂ Emission' else strongest.label_b} "
            f"given its strong statistical relationship with SO₂ emissions."
        )

    if model.quality_label in ("Weak", "Acceptable"):
        recs.append("Collect additional process variables (e.g. catalyst temperature, feed gas composition) to improve predictive accuracy.")

    if score.score >= 85:
        recs.append("Maintain current best practices — production is operating at an excellent level.")

    # De-duplicate while preserving order
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            unique_recs.append(r)
            seen.add(r)
    return unique_recs
