"""
correlation.py
==============
Automatic Pearson correlation analysis between process variables,
with plain-language industrial explanations generated for every pair.
The engineer never has to read a correlation matrix manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from config import COL_SO2, COL_DP, COL_TC, CORRELATION_BANDS
from utils import safe_run


LABELS = {
    COL_SO2: "SO₂ Emission",
    COL_DP: "Pressure Drop (ΔP)",
    COL_TC: "Global Conversion Rate (TC)",
}


@dataclass
class PairCorrelation:
    var_a: str
    var_b: str
    label_a: str
    label_b: str
    coefficient: float
    strength: str          # Negligible / Weak / Moderate / Strong / Very strong
    direction: str          # positive / negative
    interpretation: str


@dataclass
class CorrelationReport:
    matrix: pd.DataFrame
    pairs: List[PairCorrelation] = field(default_factory=list)


def _strength_label(abs_coef: float) -> str:
    for threshold, label in CORRELATION_BANDS:
        if abs_coef >= threshold:
            return label
    return "Negligible"


def _interpret_pair(label_a: str, label_b: str, coef: float, strength: str) -> str:
    direction = "positive" if coef >= 0 else "negative"
    abs_coef = abs(coef)

    if strength == "Negligible":
        return (
            f"The correlation between {label_a} and {label_b} is negligible (r = {coef:.2f}). "
            f"This means {label_b} has little to no measurable influence on {label_a}."
        )
    if strength == "Weak":
        return (
            f"The correlation between {label_a} and {label_b} is weak (r = {coef:.2f}). "
            f"This means {label_b} has limited influence on {label_a}; other factors likely dominate."
        )
    if strength == "Moderate":
        relation = "increases" if direction == "positive" else "decreases"
        return (
            f"The correlation between {label_a} and {label_b} is moderate (r = {coef:.2f}). "
            f"As {label_b} increases, {label_a} tends to {relation} — a noticeable but not dominant relationship."
        )
    if strength == "Strong":
        relation = "increases" if direction == "positive" else "decreases"
        return (
            f"The correlation between {label_a} and {label_b} is strong (r = {coef:.2f}). "
            f"As {label_b} increases, {label_a} consistently tends to {relation}, indicating "
            f"{label_b} is a significant driver worth monitoring closely."
        )
    # Very strong
    relation = "increases" if direction == "positive" else "decreases"
    return (
        f"The correlation between {label_a} and {label_b} is very strong (r = {coef:.2f}). "
        f"{label_b} is a dominant driver of {label_a}: when it increases, {label_a} reliably {relation}. "
        "This relationship should be a primary focus for process control."
    )


@safe_run("An error occurred while computing correlation analysis.")
def compute_correlations(df: pd.DataFrame) -> CorrelationReport:
    """
    Compute the full Pearson correlation matrix between SO2, delta_P
    and TC_global, and generate an automatic explanation for every pair.
    """
    cols = [COL_SO2, COL_DP, COL_TC]
    numeric_df = df[cols].apply(pd.to_numeric, errors="coerce")
    matrix = numeric_df.corr(method="pearson")

    pairs: List[PairCorrelation] = []
    seen: set[Tuple[str, str]] = set()

    for var_a in cols:
        for var_b in cols:
            if var_a == var_b:
                continue
            key = tuple(sorted((var_a, var_b)))
            if key in seen:
                continue
            seen.add(key)

            coef = float(matrix.loc[var_a, var_b])
            strength = _strength_label(abs(coef))
            direction = "positive" if coef >= 0 else "negative"
            label_a, label_b = LABELS[var_a], LABELS[var_b]

            pairs.append(PairCorrelation(
                var_a=var_a, var_b=var_b,
                label_a=label_a, label_b=label_b,
                coefficient=coef,
                strength=strength,
                direction=direction,
                interpretation=_interpret_pair(label_a, label_b, coef, strength),
            ))

    # Sort by absolute strength, most influential first
    pairs.sort(key=lambda p: abs(p.coefficient), reverse=True)

    return CorrelationReport(matrix=matrix, pairs=pairs)


def dominant_driver_of_so2(report: CorrelationReport) -> str:
    """Identify which variable most influences SO2, for the AI Decision Assistant."""
    so2_pairs = [p for p in report.pairs if COL_SO2 in (p.var_a, p.var_b)]
    if not so2_pairs:
        return "No dominant driver could be identified."
    strongest = max(so2_pairs, key=lambda p: abs(p.coefficient))
    other_label = strongest.label_b if strongest.var_a == COL_SO2 else strongest.label_a
    return f"{other_label} (r = {strongest.coefficient:.2f}, {strongest.strength.lower()})"
