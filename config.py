"""
config.py
=========
Central configuration module for the OCP Sulfuric Acid Production
AI Decision Support System (Ligne H).

This module is the single source of truth for:
    - Branding & UI color palette
    - File system paths
    - Industrial thresholds (SO2, Pressure Drop, Conversion, TC)
    - Machine Learning configuration
    - Alert / scoring configuration
    - Column mapping & synonyms used to auto-detect dataset columns

Every other module imports from here. Nothing industrial-specific
should be hard-coded elsewhere in the codebase.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


# ======================================================================
# 1. PROJECT PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"
DATA_DIR = BASE_DIR / "data"

# Sub-folders used by the Operation Log / Analysis History feature
HISTORY_DIR = DATA_DIR / "history"
LOGS_DIR = DATA_DIR / "logs"

for _p in (ASSETS_DIR, MODELS_DIR, EXPORTS_DIR, DATA_DIR, HISTORY_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

LOGO_PATH = ASSETS_DIR / "ocp_logo.png"
HISTORY_DB = HISTORY_DIR / "analysis_history.json"
OPERATION_LOG_DB = LOGS_DIR / "operation_log.json"
MODEL_ARTIFACT_PATH = MODELS_DIR / "so2_regression_model.pkl"


# ======================================================================
# 2. APPLICATION METADATA
# ======================================================================

APP_NAME = "OCP Intelligent Process Analytics"
APP_SUBTITLE = "Sulfuric Acid Production \u2014 Ligne H | AI Decision Support System"
APP_VERSION = "1.0.0"
ORGANIZATION = "OCP Group \u2014 Safi Industrial Complex (Morocco Phosphate 1)"
COPYRIGHT_LINE = f"\u00a9 {ORGANIZATION} \u2014 Internal Industrial Use Only"


# ======================================================================
# 3. OFFICIAL OCP COLOR PALETTE
# ======================================================================

class Colors:
    """Official OCP-inspired color palette used across the whole UI."""

    # Core brand colors
    DARK_BLUE = "#0B2545"       # Primary brand / header / sidebar
    DARK_BLUE_2 = "#13315C"     # Secondary depth layer
    OCP_GREEN = "#00A651"       # Primary accent / positive state
    OCP_GREEN_DARK = "#007A3D"  # Hover / pressed state
    WHITE = "#FFFFFF"
    LIGHT_GRAY = "#F4F6F9"      # App background
    MID_GRAY = "#E4E8EE"        # Card borders / dividers
    TEXT_DARK = "#101826"       # Primary text (on light backgrounds)
    TEXT_MUTED = "#5B6B82"      # Secondary text

    # Status / alert colors
    STATUS_GREEN = "#1DB954"
    STATUS_ORANGE = "#F5A623"
    STATUS_RED = "#E5484D"

    # Chart palette (harmonious, colorblind-conscious, on-brand)
    CHART_SEQUENCE: List[str] = [
        "#0B2545",  # dark blue
        "#00A651",  # ocp green
        "#F5A623",  # amber
        "#5B9BD5",  # steel blue
        "#E5484D",  # red
        "#8E7CC3",  # muted violet
        "#13315C",
        "#66C08A",
    ]

    HEATMAP_SCALE = [
        [0.0, "#E5484D"],
        [0.5, "#F4F6F9"],
        [1.0, "#00A651"],
    ]

    GLASS_BG = "rgba(255, 255, 255, 0.55)"
    GLASS_BORDER = "rgba(255, 255, 255, 0.35)"


# ======================================================================
# 4. COLUMN MAPPING / SYNONYM DETECTION
# ======================================================================
# The engineer's raw file will rarely use perfectly standardized column
# names. This dictionary lets preprocessing.py auto-detect the right
# columns regardless of naming variations (French/English, units, etc.)

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "date": [
        "date", "datetime", "timestamp", "time", "jour", "date_heure"
    ],
    "so2": [
        "so2", "so_2", "so2_emission", "emission_so2", "so2 (mg/nm3)",
        "so2_mg_nm3", "so2_ppm", "so2_stack", "so2_sortie"
    ],
    "delta_p": [
        "dp", "delta_p", "deltap", "pressure_drop", "perte_de_charge",
        "pdc", "\u0394p", "delta p", "pression_differentielle"
    ],
    "tc_global": [
        "tc", "tc_global", "conversion_rate", "taux_de_conversion",
        "conversion", "global_conversion", "tc global"
    ],
}

# Canonical (standardized) column names used internally after preprocessing
COL_DATE = "date"
COL_SO2 = "SO2"
COL_DP = "delta_P"
COL_TC = "TC_global"

REQUIRED_COLUMNS = [COL_SO2, COL_DP, COL_TC]


# ======================================================================
# 5. INDUSTRIAL THRESHOLDS
# ======================================================================
# These thresholds define what is considered Normal / Warning / Critical
# for each process variable. They are used by the Alert System, the
# Industrial Performance Score, and the AI interpretation engine.
#
# NOTE: Values are representative industrial defaults for a sulfuric
# acid double-contact/double-absorption unit. Plant engineers can
# recalibrate them in this single location.

@dataclass
class VariableThresholds:
    name: str
    unit: str
    normal_max: float
    warning_max: float
    # Above warning_max => critical
    lower_is_better: bool = True  # True if lower values = healthier process


THRESHOLDS: Dict[str, VariableThresholds] = {
    COL_SO2: VariableThresholds(
        name="SO2 Stack Emission",
        unit="mg/Nm3",
        normal_max=250.0,
        warning_max=400.0,
        lower_is_better=True,
    ),
    COL_DP: VariableThresholds(
        name="Pressure Drop (\u0394P)",
        unit="mbar",
        normal_max=120.0,
        warning_max=160.0,
        lower_is_better=True,
    ),
    COL_TC: VariableThresholds(
        name="Global Conversion Rate (TC)",
        unit="%",
        normal_max=99.8,      # for conversion, "normal_max" acts as the
        warning_max=99.5,     # minimum healthy bound (higher is better)
        lower_is_better=False,
    ),
}

# Environmental regulatory reference (typical industrial stack limit)
SO2_REGULATORY_LIMIT = 500.0  # mg/Nm3


# ======================================================================
# 6. MACHINE LEARNING CONFIGURATION
# ======================================================================

ML_TARGET = COL_SO2
ML_FEATURES = [COL_DP, COL_TC]

ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "min_rows_required": 15,   # minimum observations to train a reliable model
}

# R2 thresholds used to qualify model performance in plain language
MODEL_QUALITY_BANDS = [
    (0.85, "Excellent"),
    (0.70, "Good"),
    (0.50, "Acceptable"),
    (0.0, "Weak"),
]


# ======================================================================
# 7. CORRELATION INTERPRETATION BANDS
# ======================================================================

CORRELATION_BANDS = [
    (0.80, "Very strong"),
    (0.60, "Strong"),
    (0.40, "Moderate"),
    (0.20, "Weak"),
    (0.0, "Negligible"),
]


# ======================================================================
# 8. INDUSTRIAL PERFORMANCE SCORE WEIGHTS
# ======================================================================
# Weighted contribution of each factor to the 0-100 health score.

PERFORMANCE_SCORE_WEIGHTS = {
    "so2": 0.35,
    "delta_p": 0.20,
    "tc_global": 0.30,
    "model_confidence": 0.15,
}

SCORE_BANDS = [
    (85, "Excellent", Colors.STATUS_GREEN),
    (70, "Good", Colors.STATUS_GREEN),
    (50, "Acceptable", Colors.STATUS_ORANGE),
    (0, "Poor", Colors.STATUS_RED),
]


# ======================================================================
# 9. PREDICTIVE MAINTENANCE RULE BASE
# ======================================================================
# Maps root-cause conditions to recommended maintenance actions.
# Used by recommendation.py

MAINTENANCE_RULES = [
    {
        "condition": "so2_critical",
        "action": "Inspect catalyst activity (possible deactivation or poisoning).",
        "priority": "High",
    },
    {
        "condition": "so2_critical",
        "action": "Inspect converter beds for channeling or by-pass.",
        "priority": "High",
    },
    {
        "condition": "dp_critical",
        "action": "Inspect gas blower and ducts for fouling or mechanical wear.",
        "priority": "High",
    },
    {
        "condition": "dp_warning",
        "action": "Schedule inspection of filters/catalyst bed for clogging.",
        "priority": "Medium",
    },
    {
        "condition": "tc_critical",
        "action": "Inspect catalyst bed temperature profile and heat exchangers.",
        "priority": "High",
    },
    {
        "condition": "tc_warning",
        "action": "Check gas flow distribution and furnace combustion conditions.",
        "priority": "Medium",
    },
    {
        "condition": "model_low_confidence",
        "action": "Increase data collection frequency to improve predictive reliability.",
        "priority": "Low",
    },
]


# ======================================================================
# 10. FILE UPLOAD / SECURITY CONSTRAINTS
# ======================================================================

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
MAX_FILE_SIZE_MB = 200
MAX_ROWS_ALLOWED = 2_000_000
MAX_COLUMNS_ALLOWED = 200


# ======================================================================
# 11. STREAMLIT PAGE CONFIG (imported directly by app.py)
# ======================================================================

STREAMLIT_PAGE_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "\U0001F3ED",  # factory emoji as fallback icon
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
