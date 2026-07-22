"""
dashboard.py
============
Rendering layer for the premium industrial dashboard: KPI cards,
alert banners, the Industrial Performance Score gauge, and other
reusable Streamlit UI components. Keeping rendering separate from
business logic (statistics.py, recommendation.py, etc.) keeps the
codebase modular and testable.
"""

from __future__ import annotations

from typing import List

import streamlit as st
import plotly.graph_objects as go

from config import Colors, COL_SO2, COL_DP, COL_TC
from statistics import StatisticsReport
from prediction import RegressionResult
from recommendation import PerformanceScore, Alert, DecisionAssistantSummary
from utils import fmt_number, fmt_percent


# ======================================================================
# 1. KPI CARD GRID
# ======================================================================

def render_kpi_cards(stats: StatisticsReport, model: RegressionResult, score: PerformanceScore) -> None:
    """Render the top row of premium glassmorphism KPI cards."""
    so2, dp, tc = stats.variables[COL_SO2], stats.variables[COL_DP], stats.variables[COL_TC]

    status_color = {"Normal": Colors.STATUS_GREEN, "Warning": Colors.STATUS_ORANGE, "Critical": Colors.STATUS_RED}

    cards = [
        ("Average SO₂", f"{fmt_number(so2.mean, 1)} mg/Nm³", so2.status, status_color[so2.status], "🟢" if so2.status=="Normal" else ("🟠" if so2.status=="Warning" else "🔴")),
        ("Average ΔP", f"{fmt_number(dp.mean, 1)} mbar", dp.status, status_color[dp.status], "⚙️"),
        ("Average TC Global", f"{fmt_number(tc.mean, 2)}%", tc.status, status_color[tc.status], "🔄"),
        ("Max SO₂", f"{fmt_number(so2.maximum, 1)} mg/Nm³", "", Colors.DARK_BLUE, "📈"),
        ("Min SO₂", f"{fmt_number(so2.minimum, 1)} mg/Nm³", "", Colors.DARK_BLUE, "📉"),
        ("Observations", f"{stats.n_observations:,}", "", Colors.DARK_BLUE, "🗂️"),
        ("Prediction Accuracy", fmt_percent(model.accuracy_pct, 1), model.quality_label, Colors.OCP_GREEN, "🎯"),
        ("Performance Score", f"{score.score}/100", score.band_label, score.band_color, "🏭"),
    ]

    cols = st.columns(4)
    for idx, (title, value, badge, color, icon) in enumerate(cards):
        with cols[idx % 4]:
            _render_single_card(title, value, badge, color, icon)
        if idx % 4 == 3 and idx != len(cards) - 1:
            cols = st.columns(4)


def _render_single_card(title: str, value: str, badge: str, color: str, icon: str) -> None:
    badge_html = f'<span class="kpi-badge" style="background:{color}22;color:{color};">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-title">{title}</div>
            <div class="kpi-value" style="color:{Colors.DARK_BLUE};">{value}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
# 2. ALERT BANNERS
# ======================================================================

def render_alerts(alerts: List[Alert]) -> None:
    st.markdown('<div class="section-title">🚨 Intelligent Alert System</div>', unsafe_allow_html=True)
    if not alerts:
        st.info("No alerts generated.")
        return

    for alert in alerts:
        st.markdown(
            f"""
            <div class="alert-card" style="border-left: 6px solid {alert.color};">
                <div class="alert-header">{alert.icon} <b>{alert.title}</b></div>
                <div class="alert-message">{alert.message}</div>
                <div class="alert-action">👉 <i>{alert.recommendation}</i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ======================================================================
# 3. PERFORMANCE SCORE GAUGE
# ======================================================================

def render_score_gauge(score: PerformanceScore) -> None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score.score,
        number={"suffix": "/100", "font": {"size": 36, "color": Colors.DARK_BLUE}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": Colors.TEXT_MUTED},
            "bar": {"color": score.band_color},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#FCE8E9"},
                {"range": [50, 70], "color": "#FEF3E2"},
                {"range": [70, 100], "color": "#E6F7EE"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div class="score-band" style="color:{score.band_color};">{score.band_label}</div>', unsafe_allow_html=True)
    st.caption(score.explanation)


# ======================================================================
# 4. AI DECISION ASSISTANT PANEL
# ======================================================================

def render_decision_assistant(decision: DecisionAssistantSummary) -> None:
    st.markdown('<div class="section-title">🤖 AI Decision Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="decision-card" style="border-left: 6px solid {decision.status_color};">
            <div class="decision-status" style="color:{decision.status_color};">
                Production Status: {decision.status}
            </div>
            <p>{decision.reasoning}</p>
            <p><b>Dominant driver of SO₂:</b> {decision.dominant_driver}</p>
            <p><b>Equipment to inspect:</b> {', '.join(decision.equipment_to_inspect)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if decision.actions:
        st.markdown("**Recommended actions:**")
        for action in decision.actions:
            st.markdown(f"- {action}")


# ======================================================================
# 5. CHART + EXPLANATION HELPER (used across the whole app)
# ======================================================================

def render_chart_with_explanation(fig, explanation: str, key: str | None = None) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.markdown(f'<div class="chart-explanation">💡 {explanation}</div>', unsafe_allow_html=True)


# ======================================================================
# 6. SECTION HEADER HELPER
# ======================================================================

def section_header(title: str, icon: str = "") -> None:
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)
