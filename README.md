# ⚗️ OCP Intelligent Process Analytics
### AI Decision Support System — Sulfuric Acid Production, Ligne H

An industrial-grade decision support platform for OCP process engineers.
Upload a production dataset, click **Analyze**, and receive a complete,
automatically interpreted engineering analysis — statistics, correlations,
a predictive SO₂ model, alerts, an AI decision assistant, a performance
score, predictive maintenance suggestions, and a downloadable technical
report. No statistics, programming, or ML knowledge required.

---

## 1. Engineer Workflow

1. Open the application.
2. Upload a production dataset (`.csv`, `.xlsx`, or `.xls`) in the sidebar.
3. Click **🚀 Analyze**.

Everything else — cleaning, statistics, correlation, machine learning,
visualization, alerting, scoring, and reporting — is fully automated.

## 2. Required Data

The dataset must contain (column names are auto-detected, French or
English, various naming conventions supported):

| Logical variable | Examples of accepted column names |
|---|---|
| SO₂ emission | `SO2`, `so2_emission`, `SO2 (mg/Nm3)` |
| Pressure Drop | `DP`, `delta_p`, `perte_de_charge`, `ΔP` |
| Global Conversion Rate | `TC`, `TC_global`, `taux_de_conversion` |
| Date / Timestamp (optional) | `date`, `datetime`, `timestamp` |

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Running the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 5. Project Structure

```
ocp_dss/
├── app.py                # Main entry point — UI, orchestration, CSS
├── dashboard.py           # KPI cards, alert banners, score gauge (rendering layer)
├── preprocessing.py       # Ingestion, validation, cleaning, ML-ready prep
├── statistics.py          # Automatic descriptive statistics + interpretation
├── correlation.py         # Pearson correlation analysis + interpretation
├── prediction.py          # Multiple Linear Regression model (SO2 ~ ΔP + TC)
├── visualization.py        # All Plotly charts, each with auto-explanation
├── recommendation.py       # Alerts, performance score, AI decision assistant,
│                            predictive maintenance, recommendations
├── report.py               # Executive summary + full PDF report + exports
├── utils.py                 # Logging, safe error handling, formatting helpers
├── config.py                # Colors, thresholds, paths, ML config, scoring weights
├── requirements.txt
├── README.md
├── assets/                  # Logo & static assets
├── models/                  # Persisted trained model artifacts
├── exports/                  # Generated reports / exports (runtime)
└── data/
    ├── history/              # Analysis History store (JSON)
    └── logs/                 # Operation Log store (JSON)
```

## 6. Key Features

- **Automatic data preparation** — format detection, column auto-mapping,
  date parsing, missing-value handling, duplicate removal, outlier flagging.
- **Automatic statistics** with plain-language industrial interpretation
  for every KPI.
- **Automatic Pearson correlation analysis** with heatmap and per-pair
  plain-language explanations.
- **Machine Learning**: Multiple Linear Regression (SO₂ ~ ΔP + TC_global),
  auto-evaluated (R², RMSE, MAE) and explained in industrial terms.
- **Intelligent Alert System**: 🟢 Normal / 🟠 Warning / 🔴 Critical, with
  an immediate recommended action per alert.
- **AI Decision Assistant**: overall production status (Stable / Attention
  Needed / Critical), dominant driver of SO₂, equipment to inspect.
- **Industrial Performance Score (0–100)**: weighted health indicator
  combining SO₂, ΔP, TC, and model confidence.
- **Predictive Maintenance suggestions** driven by a condition→action rule base.
- **Smart Executive Summary** and full multi-section **PDF report**.
- **Analysis History** and **Operation Log** for traceability and audit.
- **Exports**: prediction table (CSV/Excel), full technical report (PDF).

## 7. Calibration

All industrial thresholds (SO₂, ΔP, TC), scoring weights, and maintenance
rules are centralized in `config.py` and should be reviewed/calibrated by
Ligne H process engineers to match plant-specific operating targets.

## 8. Security Notes

- Only `.csv`, `.xlsx`, `.xls` are accepted; file size and row/column
  limits are enforced (`config.py`).
- All processing errors are caught and logged internally
  (`data/logs/error_log.txt`) — the engineer only ever sees a clean,
  non-technical message.

---
© OCP Group — Internal Industrial Use Only — Ligne H, Sulfuric Acid Production
