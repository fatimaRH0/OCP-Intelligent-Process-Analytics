"""
report.py
=========
Automatic generation of:
    - A one-page Smart Executive Summary (Markdown/text, shown in-app)
    - A full multi-section technical PDF report
    - Export helpers for CSV / Excel prediction tables and PNG dashboards

Uses ReportLab for PDF generation (no external binary dependencies).
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import List

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors as rl_colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from config import APP_NAME, ORGANIZATION, Colors, COL_SO2, COL_DP, COL_TC
from statistics import StatisticsReport
from correlation import CorrelationReport
from prediction import RegressionResult
from recommendation import PerformanceScore, DecisionAssistantSummary, Alert


# ======================================================================
# 1. SMART EXECUTIVE SUMMARY (in-app, markdown-style text)
# ======================================================================

def build_executive_summary(
    stats: StatisticsReport,
    correlation: CorrelationReport,
    model: RegressionResult,
    score: PerformanceScore,
    decision: DecisionAssistantSummary,
    alerts: List[Alert],
    recommendations: List[str],
    file_name: str = "",
) -> str:
    """Return a one-page executive summary as Markdown text."""
    n_critical = sum(1 for a in alerts if a.level == "Critical")
    n_warning = sum(1 for a in alerts if a.level == "Warning")

    lines = [
        f"### Executive Summary — {APP_NAME}",
        f"**Line:** Sulfuric Acid Production — Ligne H &nbsp;&nbsp; "
        f"**Dataset:** {file_name or 'Uploaded file'} &nbsp;&nbsp; "
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Production Status: {decision.status}**",
        "",
        decision.reasoning,
        "",
        f"**Industrial Performance Score:** {score.score}/100 ({score.band_label})",
        "",
        "**Key Findings**",
        f"- Average SO₂: {stats.variables[COL_SO2].mean:.1f} — {stats.variables[COL_SO2].status}",
        f"- Average ΔP: {stats.variables[COL_DP].mean:.1f} — {stats.variables[COL_DP].status}",
        f"- Average TC Global: {stats.variables[COL_TC].mean:.2f}% — {stats.variables[COL_TC].status}",
        f"- Alerts: {n_critical} Critical, {n_warning} Warning",
        "",
        "**Model Performance**",
        f"- Predictive accuracy: {model.accuracy_pct:.1f}% (R² = {model.r2:.2f}, quality: {model.quality_label})",
        f"- Equation: {model.equation_text}",
        "",
        "**Recommendations**",
    ]
    lines += [f"- {rec}" for rec in recommendations[:6]]
    lines += [
        "",
        "**Conclusion**",
        (
            f"Ligne H is currently classified as **{decision.status}**. "
            + ("Immediate action is advised on the flagged variables above." if decision.status == "Critical"
               else "Preventive follow-up is advised on the flagged variables above." if decision.status == "Attention Needed"
               else "No corrective action is required; continue standard monitoring.")
        ),
    ]
    return "\n".join(lines)


# ======================================================================
# 2. FULL TECHNICAL PDF REPORT
# ======================================================================

def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="OCPTitle", fontSize=20, leading=24,
                               textColor=rl_colors.HexColor(Colors.DARK_BLUE), spaceAfter=6))
    styles.add(ParagraphStyle(name="OCPSubtitle", fontSize=11, leading=14,
                               textColor=rl_colors.HexColor(Colors.TEXT_MUTED), spaceAfter=14))
    styles.add(ParagraphStyle(name="OCPSection", fontSize=14, leading=18,
                               textColor=rl_colors.HexColor(Colors.DARK_BLUE), spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="OCPBody", fontSize=10, leading=15,
                               textColor=rl_colors.HexColor(Colors.TEXT_DARK), alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="OCPBullet", fontSize=10, leading=15, leftIndent=12,
                               textColor=rl_colors.HexColor(Colors.TEXT_DARK)))
    return styles


def generate_pdf_report(
    stats: StatisticsReport,
    correlation: CorrelationReport,
    model: RegressionResult,
    score: PerformanceScore,
    decision: DecisionAssistantSummary,
    alerts: List[Alert],
    recommendations: List[str],
    maintenance: List[dict],
    file_name: str = "",
) -> bytes:
    """Build the complete multi-section technical PDF report and return raw bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = _pdf_styles()
    story = []

    # ---- Cover / Header ----
    story.append(Paragraph(APP_NAME, styles["OCPTitle"]))
    story.append(Paragraph(f"{ORGANIZATION} — Sulfuric Acid Production, Ligne H", styles["OCPSubtitle"]))
    story.append(Paragraph(
        f"Dataset: {file_name or 'N/A'} &nbsp;|&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"&nbsp;|&nbsp; Observations: {stats.n_observations}",
        styles["OCPBody"]
    ))
    story.append(Spacer(1, 12))

    # ---- Executive Summary ----
    story.append(Paragraph("1. Executive Summary", styles["OCPSection"]))
    story.append(Paragraph(f"<b>Production Status:</b> {decision.status}", styles["OCPBody"]))
    story.append(Paragraph(decision.reasoning, styles["OCPBody"]))
    story.append(Paragraph(f"<b>Industrial Performance Score:</b> {score.score}/100 ({score.band_label})", styles["OCPBody"]))

    # ---- Statistical Analysis ----
    story.append(Paragraph("2. Statistical Analysis", styles["OCPSection"]))
    table_data = [["Variable", "Mean", "Median", "Std Dev", "Min", "Max", "Status"]]
    for var in stats.variables.values():
        table_data.append([
            var.label, f"{var.mean:.2f}", f"{var.median:.2f}", f"{var.std:.2f}",
            f"{var.minimum:.2f}", f"{var.maximum:.2f}", var.status,
        ])
    story.append(_styled_table(table_data))
    for var in stats.variables.values():
        story.append(Paragraph(f"<b>{var.label}:</b> {var.interpretation}", styles["OCPBullet"]))

    # ---- Correlation Analysis ----
    story.append(Paragraph("3. Correlation Analysis", styles["OCPSection"]))
    corr_table = [["Variable A", "Variable B", "Coefficient (r)", "Strength"]]
    for p in correlation.pairs:
        corr_table.append([p.label_a, p.label_b, f"{p.coefficient:.2f}", p.strength])
    story.append(_styled_table(corr_table))
    for p in correlation.pairs:
        story.append(Paragraph(f"• {p.interpretation}", styles["OCPBullet"]))

    # ---- Machine Learning Results ----
    story.append(Paragraph("4. Machine Learning Results", styles["OCPSection"]))
    story.append(Paragraph(f"<b>Model:</b> Multiple Linear Regression (SO₂ ~ ΔP + TC_global)", styles["OCPBody"]))
    story.append(Paragraph(f"<b>Equation:</b> {model.equation_text}", styles["OCPBody"]))
    ml_table = [
        ["R²", "RMSE", "MAE", "Quality", "Approx. Accuracy"],
        [f"{model.r2:.3f}", f"{model.rmse:.2f}", f"{model.mae:.2f}", model.quality_label, f"{model.accuracy_pct:.1f}%"],
    ]
    story.append(_styled_table(ml_table))
    story.append(Paragraph(model.interpretation, styles["OCPBody"]))

    # ---- Alerts ----
    story.append(Paragraph("5. Alerts & Immediate Recommendations", styles["OCPSection"]))
    for alert in alerts:
        story.append(Paragraph(f"{alert.icon} <b>{alert.title}</b> — {alert.message}", styles["OCPBullet"]))
        story.append(Paragraph(f"<i>Action: {alert.recommendation}</i>", styles["OCPBullet"]))

    # ---- Predictive Maintenance ----
    story.append(Paragraph("6. Predictive Maintenance Suggestions", styles["OCPSection"]))
    for m in maintenance:
        story.append(Paragraph(f"• [{m['priority']}] {m['action']}", styles["OCPBullet"]))

    # ---- Recommendations ----
    story.append(Paragraph("7. Recommendations", styles["OCPSection"]))
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles["OCPBullet"]))

    # ---- Conclusion ----
    story.append(Paragraph("8. Conclusion", styles["OCPSection"]))
    conclusion = (
        f"Ligne H is currently classified as <b>{decision.status}</b>. "
        + ("Immediate corrective action is required on the flagged variables above."
           if decision.status == "Critical"
           else "Preventive follow-up is advised on the flagged variables above."
           if decision.status == "Attention Needed"
           else "No corrective action is required; continue standard monitoring.")
    )
    story.append(Paragraph(conclusion, styles["OCPBody"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _styled_table(data: List[List[str]]) -> Table:
    table = Table(data, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(Colors.DARK_BLUE)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor(Colors.MID_GRAY)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor(Colors.LIGHT_GRAY)]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


# ======================================================================
# 3. DATA EXPORT HELPERS
# ======================================================================

def export_predictions_to_csv(prediction_table: pd.DataFrame) -> bytes:
    return prediction_table.to_csv(index=False).encode("utf-8")


def export_predictions_to_excel(prediction_table: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        prediction_table.to_excel(writer, index=False, sheet_name="SO2_Predictions")
    buffer.seek(0)
    return buffer.getvalue()
