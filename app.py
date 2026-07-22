"""
app.py
======
Main entry point of the OCP Intelligent Process Analytics platform.

Engineer workflow (by design, only three actions):
    1. Open the application
    2. Upload a production dataset (CSV / Excel)
    3. Click "Analyze"

Everything else — cleaning, statistics, correlation, machine learning,
visualization, alerting, scoring, recommendations, and reporting — is
fully automated by the modules this file orchestrates.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime

from config import (
    STREAMLIT_PAGE_CONFIG, APP_NAME, APP_SUBTITLE, APP_VERSION, COPYRIGHT_LINE,
    LOGO_PATH, Colors, COL_SO2, COL_DP, COL_TC, COL_DATE,
)
from utils import AppError, log_operation, save_analysis_to_history, load_analysis_history, get_operation_log
from preprocessing import load_file, preprocess_dataset
from statistics import compute_statistics, overall_variability_comment
from correlation import compute_correlations
from prediction import train_regression_model, build_prediction_table, forecast_next_risk
import visualization as viz
from recommendation import (
    generate_alerts, compute_performance_score, build_decision_assistant,
    predictive_maintenance_suggestions, generate_recommendations,
)
from report import build_executive_summary, generate_pdf_report, export_predictions_to_csv, export_predictions_to_excel
import dashboard as dash


# ======================================================================
# PAGE CONFIG (must be first Streamlit call)
# ======================================================================

st.set_page_config(**STREAMLIT_PAGE_CONFIG)


# ======================================================================
# PREMIUM INDUSTRIAL CSS (glassmorphism, OCP branding)
# ======================================================================

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(180deg, {Colors.OCP_GREEN} 0%, #EAEFF5 100%);
        }}

        /* ---- Header ---- */
        .app-header {{
            background: linear-gradient(135deg, {Colors.DARK_BLUE} 0%, {Colors.DARK_BLUE_2} 100%);
            padding: 28px 36px;
            border-radius: 18px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(11,37,69,0.25);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .app-title {{
            color: white;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 0.3px;
            margin: 0;
        }}
        .app-subtitle {{
            color: #C7D3E8;
            font-size: 14px;
            font-weight: 500;
            margin-top: 4px;
        }}
        .app-badge {{
            background: {Colors.OCP_GREEN};
            color: white;
            padding: 6px 16px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        /* ---- Section titles ---- */
        .section-title {{
            font-size: 19px;
            font-weight: 700;
            color: {Colors.DARK_BLUE};
            margin: 22px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid {Colors.OCP_GREEN}33;
        }}

        /* ---- KPI Cards (glassmorphism) ---- */
        .kpi-card {{
            background: {Colors.GLASS_BG};
            backdrop-filter: blur(10px);
            border: 1px solid {Colors.GLASS_BORDER};
            border-radius: 16px;
            padding: 18px 18px 14px 18px;
            margin-bottom: 16px;
            box-shadow: 0 4px 14px rgba(11,37,69,0.08);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 24px rgba(11,37,69,0.15);
        }}
        .kpi-icon {{ font-size: 22px; margin-bottom: 4px; }}
        .kpi-title {{ font-size: 12.5px; font-weight: 600; color: {Colors.TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.4px; }}
        .kpi-value {{ font-size: 24px; font-weight: 800; margin-top: 4px; }}
        .kpi-badge {{ display: inline-block; margin-top: 8px; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; }}

        /* ---- Alert cards ---- */
        .alert-card {{
            background: {Colors.OCP_GREEN}11;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(11,37,69,0.06);
        }}
        .alert-header {{ font-size: 15px; color: {Colors.TEXT_DARK}; margin-bottom: 4px; }}
        .alert-message {{ font-size: 13.5px; color: {Colors.TEXT_MUTED}; }}
        .alert-action {{ font-size: 13px; color: {Colors.DARK_BLUE}; margin-top: 6px; }}

        /* ---- Decision assistant card ---- */
        .decision-card {{
            background: {Colors.OCP_GREEN}11;
            border-radius: 14px;
            padding: 20px 22px;
            box-shadow: 0 4px 14px rgba(11,37,69,0.08);
        }}
        .decision-status {{ font-size: 20px; font-weight: 800; margin-bottom: 10px; }}

        /* ---- Score band label ---- */
        .score-band {{ text-align: center; font-size: 20px; font-weight: 800; margin-top: -10px; }}

        /* ---- Chart explanation caption ---- */
        .chart-explanation {{
            background: {Colors.OCP_GREEN}11;
            border-left: 4px solid {Colors.OCP_GREEN};
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 13.5px;
            color: {Colors.TEXT_DARK};
            margin-bottom: 22px;
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {Colors.DARK_BLUE} 0%, {Colors.DARK_BLUE_2} 100%);
        }}
        section[data-testid="stSidebar"] * {{ color: #E8EDF5 !important; }}

        /* ---- Buttons ---- */
        .stButton > button {{
            background: linear-gradient(135deg, {Colors.OCP_GREEN} 0%, {Colors.OCP_GREEN_DARK} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 22px;
            font-weight: 700;
            letter-spacing: 0.3px;
            box-shadow: 0 4px 12px rgba(0,166,81,0.3);
            transition: all 0.15s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 18px rgba(0,166,81,0.4);
        }}

        /* ---- Footer ---- */
        .app-footer {{
            text-align: center;
            color: {Colors.TEXT_MUTED};
            font-size: 12px;
            padding: 24px 0 10px 0;
        }}

        /* Hide default Streamlit chrome for an industrial software feel */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"""
            <div class="app-header">
                <div>
                    <p class="app-title">⚗️ {APP_NAME}</p>
                    <p class="app-subtitle">{APP_SUBTITLE}</p>
                </div>
                <div class="app-badge">LIGNE H · LIVE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(f'<div class="app-footer">{COPYRIGHT_LINE} — v{APP_VERSION}</div>', unsafe_allow_html=True)


# ======================================================================
# SIDEBAR — Upload & Navigation
# ======================================================================

def render_sidebar():
    with st.sidebar:
        st.image(LOGO_PATH,width=170)
        st.markdown("## ⚗️ Control Panel")
        st.caption("Upload a production dataset to begin.")
        uploaded_file = st.file_uploader(
            "Production Dataset (CSV / Excel)",
            type=["csv", "xlsx", "xls"],
            help="The file must contain SO2, Pressure Drop (ΔP) and Global Conversion Rate (TC) columns.",
        )
        analyze_clicked = st.button("🚀 Analyze", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Go to",
            ["Dashboard", "Statistics", "Correlation", "Machine Learning",
             "Alerts & AI Assistant", "Analysis History", "Operation Log", "Executive Report"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(f"OCP Intelligence Suite v{APP_VERSION}")
        return uploaded_file, analyze_clicked, page


# ======================================================================
# CORE ANALYSIS PIPELINE
# ======================================================================

def run_full_analysis(uploaded_file):
    raw_df = load_file(uploaded_file)
    clean_df, prep_report = preprocess_dataset(raw_df)

    stats_report = compute_statistics(clean_df)
    corr_report = compute_correlations(clean_df)
    model_result = train_regression_model(clean_df)
    prediction_table = build_prediction_table(clean_df, model_result)

    slope = viz._trend_slope(clean_df[COL_SO2].values)
    forecast_note = forecast_next_risk(model_result, slope)

    alerts = generate_alerts(stats_report, forecast_note)
    score = compute_performance_score(stats_report, model_result)
    decision = build_decision_assistant(stats_report, corr_report, model_result, score)
    maintenance = predictive_maintenance_suggestions(stats_report, model_result)
    recommendations = generate_recommendations(stats_report, corr_report, model_result, score)

    return {
        "raw_df": raw_df, "clean_df": clean_df, "prep_report": prep_report,
        "stats": stats_report, "correlation": corr_report, "model": model_result,
        "prediction_table": prediction_table, "forecast_note": forecast_note,
        "alerts": alerts, "score": score, "decision": decision,
        "maintenance": maintenance, "recommendations": recommendations,
        "file_name": getattr(uploaded_file, "name", "dataset"),
    }


# ======================================================================
# PAGE RENDERERS
# ======================================================================

def page_dashboard(R: dict) -> None:
    dash.section_header("Industrial Dashboard", "🏭")
    dash.render_kpi_cards(R["stats"], R["model"], R["score"])

    col1, col2 = st.columns([1, 2])
    with col1:
        dash.section_header("Performance Score", "🎯")
        dash.render_score_gauge(R["score"])
    with col2:
        dash.section_header("SO₂ Evolution", "📈")
        fig, exp = viz.chart_evolution(R["clean_df"], COL_SO2, "SO₂ Emission")
        dash.render_chart_with_explanation(fig, exp, key="dash_evo")

    st.info(f"📊 {overall_variability_comment(R['stats'])}")


def page_statistics(R: dict) -> None:
    dash.section_header("Automatic Statistical Analysis", "📐")
    stats = R["stats"]
    for col, var in stats.variables.items():
        with st.expander(f"{var.label} — {var.status}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{var.mean:.2f}")
            c2.metric("Median", f"{var.median:.2f}")
            c3.metric("Std Dev", f"{var.std:.2f}")
            c4.metric("Variance", f"{var.variance:.2f}")
            c1.metric("Min", f"{var.minimum:.2f}")
            c2.metric("Max", f"{var.maximum:.2f}")
            c3.metric("Q1", f"{var.q1:.2f}")
            c4.metric("Q3", f"{var.q3:.2f}")
            st.markdown(f'<div class="chart-explanation">💡 {var.interpretation}</div>', unsafe_allow_html=True)

    dash.section_header("Distributions", "📊")
    for col, label in [(COL_SO2, "SO₂ Emission"), (COL_DP, "Pressure Drop"), (COL_TC, "TC Global")]:
        c1, c2 = st.columns(2)
        with c1:
            fig, exp = viz.chart_histogram(R["clean_df"], col, label)
            dash.render_chart_with_explanation(fig, exp, key=f"hist_{col}")
        with c2:
            fig, exp = viz.chart_boxplot(R["clean_df"], col, label)
            dash.render_chart_with_explanation(fig, exp, key=f"box_{col}")


def page_correlation(R: dict) -> None:
    dash.section_header("Automatic Correlation Analysis", "🔗")
    fig, exp = viz.chart_heatmap(R["correlation"].matrix)
    dash.render_chart_with_explanation(fig, exp, key="heatmap")

    st.markdown("**Correlation Table**")
    table = pd.DataFrame([{
        "Variable A": p.label_a, "Variable B": p.label_b,
        "Coefficient (r)": round(p.coefficient, 3), "Strength": p.strength,
    } for p in R["correlation"].pairs])
    st.dataframe(table, use_container_width=True, hide_index=True)

    for p in R["correlation"].pairs:
        st.markdown(f'<div class="chart-explanation">💡 {p.interpretation}</div>', unsafe_allow_html=True)

    dash.section_header("Scatter Analysis", "🎯")
    c1, c2 = st.columns(2)
    with c1:
        fig, exp = viz.chart_scatter(R["clean_df"], COL_DP, "Pressure Drop (ΔP)")
        dash.render_chart_with_explanation(fig, exp, key="scatter_dp")
    with c2:
        fig, exp = viz.chart_scatter(R["clean_df"], COL_TC, "TC Global")
        dash.render_chart_with_explanation(fig, exp, key="scatter_tc")

    dash.section_header("Trend Comparison", "📉")
    fig, exp = viz.chart_trend_overlay(R["clean_df"])
    dash.render_chart_with_explanation(fig, exp, key="trend_overlay")


def page_ml(R: dict) -> None:
    model = R["model"]
    dash.section_header("Machine Learning — SO₂ Prediction Model", "🧠")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{model.r2:.3f}")
    c2.metric("RMSE", f"{model.rmse:.2f}")
    c3.metric("MAE", f"{model.mae:.2f}")
    c4.metric("Accuracy", f"{model.accuracy_pct:.1f}%")

    st.markdown(f"**Regression Equation:** `{model.equation_text}`")
    st.markdown(f'<div class="chart-explanation">💡 {model.interpretation}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, exp = viz.chart_real_vs_predicted(
            R["clean_df"].loc[model.predictions.index, COL_SO2], model.predictions, model.r2
        )
        dash.render_chart_with_explanation(fig, exp, key="real_vs_pred")
    with c2:
        fig, exp = viz.chart_residuals(model.predictions, model.residuals)
        dash.render_chart_with_explanation(fig, exp, key="residuals")

    dash.section_header("Prediction Table", "📋")
    st.dataframe(R["prediction_table"], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Export Predictions (CSV)",
            data=export_predictions_to_csv(R["prediction_table"]),
            file_name=f"so2_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇️ Export Predictions (Excel)",
            data=export_predictions_to_excel(R["prediction_table"]),
            file_name=f"so2_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def page_alerts(R: dict) -> None:
    dash.render_alerts(R["alerts"])
    st.markdown("---")
    dash.render_decision_assistant(R["decision"])
    st.markdown("---")
    dash.section_header("Predictive Maintenance Suggestions", "🔧")
    for m in R["maintenance"]:
        priority_icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(m["priority"], "⚪")
        st.markdown(f"{priority_icon} **[{m['priority']}]** {m['action']}")


def page_history() -> None:
    dash.section_header("Analysis History", "🕒")
    history_df = load_analysis_history()
    if history_df.empty:
        st.info("No previous analyses recorded yet. Run an analysis and it will appear here.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)


def page_operation_log() -> None:
    dash.section_header("Operation Log", "📜")
    log_df = get_operation_log()
    if log_df.empty:
        st.info("No operations recorded yet.")
    else:
        st.dataframe(log_df, use_container_width=True, hide_index=True)


def page_executive_report(R: dict) -> None:
    dash.section_header("Smart Executive Summary", "📄")
    summary_text = build_executive_summary(
        R["stats"], R["correlation"], R["model"], R["score"], R["decision"],
        R["alerts"], R["recommendations"], R["file_name"],
    )
    st.markdown(summary_text)

    st.markdown("---")
    pdf_bytes = generate_pdf_report(
        R["stats"], R["correlation"], R["model"], R["score"], R["decision"],
        R["alerts"], R["recommendations"], R["maintenance"], R["file_name"],
    )
    st.download_button(
        "⬇️ Download Full Technical Report (PDF)",
        data=pdf_bytes,
        file_name=f"OCP_LigneH_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    inject_css()
    render_header()

    uploaded_file, analyze_clicked, page = render_sidebar()

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Please upload a production dataset (CSV or Excel) before clicking Analyze.")
        else:
            with st.spinner("Analyzing production data... preparing dataset, training model, generating insights."):
                try:
                    result = run_full_analysis(uploaded_file)
                    st.session_state["analysis_result"] = result
                    st.success("✅ Dataset successfully prepared and analyzed.")

                    save_analysis_to_history({
                        "file_name": result["file_name"],
                        "n_observations": result["stats"].n_observations,
                        "model_r2": round(result["model"].r2, 3),
                        "risk_level": result["decision"].status,
                    })
                except AppError as e:
                    st.error(f"⚠️ {e.message}")
                    return

    result = st.session_state.get("analysis_result")

    if result is None and page not in ("Analysis History", "Operation Log"):
        st.markdown(
            """
            <div class="decision-card" style="text-align:center; padding:60px 30px;">
                <h3>👋 Welcome to the OCP Intelligent Process Analytics platform</h3>
                <p>Upload a production dataset in the sidebar and click <b>Analyze</b> to generate a
                complete automatic engineering analysis: statistics, correlations, predictive modeling,
                alerts, and recommendations — no manual calculation required.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if page == "Dashboard":
            page_dashboard(result)
        elif page == "Statistics":
            page_statistics(result)
        elif page == "Correlation":
            page_correlation(result)
        elif page == "Machine Learning":
            page_ml(result)
        elif page == "Alerts & AI Assistant":
            page_alerts(result)
        elif page == "Analysis History":
            page_history()
        elif page == "Operation Log":
            page_operation_log()
        elif page == "Executive Report":
            page_executive_report(result)

    render_footer()


if __name__ == "__main__":
    main()
