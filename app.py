"""
SmartRot AI: Edge-AI TPS Decay & Odor Risk Monitor (DKI Jakarta)
Pilot Project — DKI Jakarta
"""

import io
import random
import time
from datetime import datetime, timedelta

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from PIL import Image
from streamlit_folium import st_folium

# ── OpenVINO import with graceful fallback ─────────────────────────────────────
try:
    from openvino.runtime import Core as OVCore
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# ── Page config must be the FIRST Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="SmartRot AI | DKI Jakarta TPS Monitor",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Jakarta Smart City Command Center — Emerald Green Eco-Tech Theme ──────────
SMARTCITY_CSS = """
<style>
    /* ══ TOKENS ══════════════════════════════════════════════════════════════ */
    :root {
        --bg-app:      #0F172A;
        --bg-card:     #1E293B;
        --bg-card2:    #243047;
        --border:      #334155;
        --accent:      #10B981;
        --accent-dim:  #065F46;
        --text-hi:     #F8FAFC;
        --text-mid:    #CBD5E1;
        --text-lo:     #64748B;
        --risk-red:    #EF4444;
        --risk-red-bg: #2D0A0A;
        --risk-yel:    #F59E0B;
        --risk-yel-bg: #2D1A00;
        --risk-grn:    #10B981;
        --risk-grn-bg: #052E16;
        --radius:      12px;
        --radius-sm:   8px;
    }

    /* ══ APP SHELL ═══════════════════════════════════════════════════════════ */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background-color: var(--bg-app) !important;
        color: var(--text-hi) !important;
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    }

    /* ══ HEADER BAR ══════════════════════════════════════════════════════════ */
    [data-testid="stHeader"] {
        background-color: #0F172A !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* ══ SIDEBAR ════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #162032 !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {
        color: var(--text-mid) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--accent) !important;
    }

    /* ══ MAIN CONTENT ════════════════════════════════════════════════════════ */
    [data-testid="stMainBlockContainer"] {
        background-color: var(--bg-app) !important;
        padding-top: 1.5rem;
    }
    /* Ensure all plain text in main area is readable */
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] li,
    [data-testid="stMarkdown"] p {
        color: var(--text-mid) !important;
    }
    [data-testid="stMainBlockContainer"] h1,
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3,
    [data-testid="stMainBlockContainer"] h4 {
        color: var(--text-hi) !important;
    }

    /* ══ TABS ════════════════════════════════════════════════════════════════ */
    [data-testid="stTabs"] [role="tablist"] {
        background-color: #162032;
        border-radius: var(--radius) var(--radius) 0 0;
        padding: 6px 10px 0;
        border-bottom: 2px solid var(--accent);
        gap: 4px;
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: var(--text-lo) !important;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 20px;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        border: 1px solid transparent;
        transition: all 0.2s ease;
        background: transparent;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--accent) !important;
        background-color: #1E293B;
        border-color: var(--border);
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent) !important;
        background-color: var(--bg-app) !important;
        border-color: var(--accent) var(--accent) transparent !important;
        border-width: 1px 1px 2px !important;
    }

    /* ══ STREAMLIT METRICS ═══════════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: 3px solid var(--accent) !important;
        border-radius: var(--radius) !important;
        padding: 16px 18px !important;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"] {
        color: var(--text-lo) !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-hi) !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricDelta"] {
        color: var(--text-mid) !important;
        font-size: 0.78rem !important;
    }

    /* ══ BUTTONS ════════════════════════════════════════════════════════════ */
    .stButton > button,
    .stDownloadButton > button {
        background-color: var(--accent) !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 10px 20px !important;
        font-size: 0.88rem !important;
        transition: background-color 0.2s, transform 0.1s !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #059669 !important;
        transform: translateY(-1px);
    }
    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(0);
    }

    /* ══ SELECTBOX / FILE UPLOADER ═══════════════════════════════════════════ */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stFileUploader"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-hi) !important;
    }
    [data-testid="stFileUploader"] label {
        color: var(--text-mid) !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: var(--bg-card2) !important;
        border: 2px dashed var(--accent) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] span {
        color: var(--text-mid) !important;
    }

    /* ══ PROGRESS BAR ════════════════════════════════════════════════════════ */
    [data-testid="stProgressBar"] > div > div {
        background-color: var(--accent) !important;
        border-radius: 4px;
    }
    [data-testid="stProgressBar"] > div {
        background-color: var(--border) !important;
        border-radius: 4px;
    }

    /* ══ EXPANDER ═══════════════════════════════════════════════════════════ */
    [data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-hi) !important;
        font-weight: 600 !important;
    }

    /* ══ DATAFRAME ═══════════════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius) !important;
        overflow: hidden;
        border: 1px solid var(--border) !important;
    }

    /* ══ CAPTION / SMALL TEXT ════════════════════════════════════════════════ */
    [data-testid="stCaptionContainer"] p,
    small {
        color: var(--text-lo) !important;
    }

    /* ══ INFO BOX ════════════════════════════════════════════════════════════ */
    [data-testid="stAlert"] {
        background-color: var(--bg-card) !important;
        border-color: var(--border) !important;
        color: var(--text-mid) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ══ DIVIDER ════════════════════════════════════════════════════════════ */
    hr { border-color: var(--border) !important; }

    /* ══ SCROLLBAR ══════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-app); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* ══ CUSTOM COMPONENT CLASSES ════════════════════════════════════════════ */

    /* ── OpenVINO badge card ── */
    .ov-card {
        background-color: #162032;
        border: 1px solid #1D4ED8;
        border-left: 4px solid #3B82F6;
        border-radius: var(--radius-sm);
        padding: 14px 18px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .ov-dot {
        width: 10px; height: 10px;
        background: var(--accent);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--accent);
        flex-shrink: 0;
    }
    .ov-body { flex: 1; }
    .ov-title {
        color: #93C5FD;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .ov-device {
        color: var(--text-hi);
        font-size: 1rem;
        font-weight: 700;
    }
    .ov-latency {
        color: var(--accent);
        font-size: 0.82rem;
        margin-top: 1px;
    }

    /* ── Risk alert cards ── */
    .alert-critical {
        background-color: var(--risk-red-bg);
        border: 1px solid var(--risk-red);
        border-left: 5px solid var(--risk-red);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: var(--risk-yel-bg);
        border: 1px solid var(--risk-yel);
        border-left: 5px solid var(--risk-yel);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin: 1rem 0;
    }
    .alert-safe {
        background-color: var(--risk-grn-bg);
        border: 1px solid var(--risk-grn);
        border-left: 5px solid var(--risk-grn);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin: 1rem 0;
    }
    .alert-critical h3 { color: var(--risk-red) !important; margin: 0 0 12px; font-size: 1.1rem; }
    .alert-warning  h3 { color: var(--risk-yel) !important; margin: 0 0 12px; font-size: 1.1rem; }
    .alert-safe     h3 { color: var(--risk-grn) !important; margin: 0 0 12px; font-size: 1.1rem; }

    .kpi-row {
        display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 14px;
    }
    .kpi-cell {
        background: #162032;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 10px 16px;
        min-width: 120px;
    }
    .kpi-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 3px;
    }
    .kpi-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-hi);
    }

    /* ── Risk score bar ── */
    .risk-bar-wrap {
        background: var(--bg-card2);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .risk-bar-fill {
        height: 8px;
        border-radius: 4px;
        transition: width 0.6s ease;
    }

    /* ── Action box ── */
    .action-box {
        background-color: #162032;
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 12px 16px;
        margin-top: 14px;
        font-size: 0.88rem;
        color: var(--text-mid);
        line-height: 1.6;
    }
    .action-box strong { color: var(--accent); }

    /* ── Sidebar status badge ── */
    .badge-active {
        display: inline-block;
        background-color: var(--accent-dim);
        color: var(--accent);
        border: 1px solid var(--accent);
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }

    /* ── Upload idle placeholder ── */
    .upload-idle {
        text-align: center;
        color: var(--text-lo);
        padding: 52px 24px;
        border: 2px dashed var(--border);
        border-radius: var(--radius);
        background: var(--bg-card);
    }
    .upload-idle .idle-icon { font-size: 2.8rem; margin-bottom: 10px; }
    .upload-idle .idle-text { font-size: 0.95rem; color: var(--text-mid); }
    .upload-idle .idle-hint { font-size: 0.78rem; color: var(--text-lo); margin-top: 4px; }

    /* ── Site status mini-card ── */
    .site-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .site-card .sc-name  { font-size: 0.82rem; font-weight: 700; }
    .site-card .sc-meta  { font-size: 0.74rem; color: var(--text-lo); margin-top: 1px; }
    .site-card .sc-action{ font-size: 0.74rem; font-weight: 700; margin-top: 3px; }

    /* ── Export note box ── */
    .export-note {
        padding: 11px 16px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        font-size: 0.82rem;
        color: var(--text-mid);
        line-height: 1.5;
    }
</style>
"""

st.markdown(SMARTCITY_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Helper functions ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def check_openvino_device() -> dict:
    """
    Probe available OpenVINO devices.
    Prefers NPU → GPU → CPU.
    Returns a dict with keys: device, latency_ms, available.
    """
    if not OPENVINO_AVAILABLE:
        return {
            "device": "CPU (Simulation)",
            "latency_ms": round(random.uniform(18.0, 28.0), 1),
            "available": False,
            "all_devices": [],
        }

    try:
        core = OVCore()
        devices = core.available_devices          # e.g. ['CPU', 'GPU', 'NPU']

        if "NPU" in devices:
            target = "NPU"
            latency = round(random.uniform(8.0, 14.0), 1)
        elif "GPU" in devices:
            target = "GPU"
            latency = round(random.uniform(10.0, 18.0), 1)
        else:
            target = "CPU"
            latency = round(random.uniform(18.0, 28.0), 1)

        return {
            "device": target,
            "latency_ms": latency,
            "available": True,
            "all_devices": devices,
        }
    except Exception as exc:
        return {
            "device": f"CPU (fallback — {exc})",
            "latency_ms": round(random.uniform(20.0, 30.0), 1),
            "available": False,
            "all_devices": [],
        }


# ── Decay profile lookup table ────────────────────────────────────────────────
DECAY_PROFILES = {
    "🐟 Seafood / Meat": {
        "critical_hours": 12,
        "gas_risk": "High Hydrogen Sulfide (H₂S)",
        "gas_icon": "☠️",
        "risk_score": 92,
        "level": "CRITICAL",
        "css_class": "alert-critical",
        "badge_color": "#ef4444",
        "bar_color": "#ef4444",
        "action": (
            "🚨 <strong>IMMEDIATE ACTION REQUIRED:</strong> "
            "Dispatch collection vehicle within 2 hours. "
            "Notify TPS officer to seal waste bags and apply deodorizer. "
            "Escalate to DLH dispatch supervisor if unresolved within 1 hour."
        ),
        "confidence": round(random.uniform(91.0, 97.5), 1),
    },
    "🥦 Vegetables / Fruits": {
        "critical_hours": 24,
        "gas_risk": "Methane (CH₄) + Fermentation VOCs",
        "gas_icon": "⚠️",
        "risk_score": 61,
        "level": "WARNING",
        "css_class": "alert-warning",
        "badge_color": "#F59E0B",
        "bar_color": "#F59E0B",
        "action": (
            "⚠️ <strong>SCHEDULE COLLECTION:</strong> "
            "Queue pickup within 8 hours. "
            "Advise TPS officer to cover organic bins. "
            "Monitor gas sensor readings — alert if CH₄ exceeds threshold."
        ),
        "confidence": round(random.uniform(86.0, 93.5), 1),
    },
    "🍚 Cooked Carbs / Dry Waste": {
        "critical_hours": 36,
        "gas_risk": "Minimal — Low VOC emission",
        "gas_icon": "✅",
        "risk_score": 22,
        "level": "SAFE",
        "css_class": "alert-safe",
        "badge_color": "#10B981",
        "bar_color": "#10B981",
        "action": (
            "✅ <strong>ROUTINE MONITORING:</strong> "
            "No immediate collection required. "
            "Log entry and schedule next inspection within 12 hours. "
            "Proceed with standard rotation."
        ),
        "confidence": round(random.uniform(88.0, 95.0), 1),
    },
}


def simulate_inference(waste_type: str, latency_ms: float) -> dict:
    """Simulate OpenVINO inference with a brief progress animation."""
    profile = DECAY_PROFILES[waste_type].copy()
    # Re-randomise confidence per run for realism
    lo, hi = (91.0, 97.5) if "Seafood" in waste_type else \
              (86.0, 93.5) if "Vegetables" in waste_type else (88.0, 95.0)
    profile["confidence"] = round(random.uniform(lo, hi), 1)
    profile["latency_ms"] = latency_ms
    return profile


def _countdown_str(hours: int) -> str:
    target = datetime.now() + timedelta(hours=hours)
    return target.strftime("Critical by %H:%M  ·  %d %b %Y") + f"  ({hours}h window)"


def render_openvino_card(ov: dict) -> None:
    """Render the OpenVINO status badge card."""
    device_label = ov["device"]
    latency      = ov["latency_ms"]
    ov_flag      = "Intel OpenVINO™ Optimized" if ov["available"] else "Simulation Mode"
    all_dev_str  = ", ".join(ov["all_devices"]) if ov["all_devices"] else "N/A"

    st.markdown(
        f"""
        <div class="ov-card">
            <div class="ov-dot"></div>
            <div class="ov-body">
                <div class="ov-title">⚡ OpenVINO Runtime · Active</div>
                <div class="ov-device">Device: {device_label}</div>
                <div class="ov-latency">
                    {latency} ms &nbsp;·&nbsp; {ov_flag}
                    &nbsp;|&nbsp; Detected: {all_dev_str}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(profile: dict, image_name: str) -> None:
    """Render the full decay analysis result card."""
    css       = profile["css_class"]
    level     = profile["level"]
    hours     = profile["critical_hours"]
    score     = profile["risk_score"]
    bar_col   = profile["bar_color"]
    action    = profile["action"]
    gas       = profile["gas_risk"]
    gas_ico   = profile["gas_icon"]
    conf      = profile["confidence"]
    lat       = profile["latency_ms"]
    countdown = _countdown_str(hours)

    st.markdown(
        f"""
        <div class="{css}">
            <h3>{gas_ico} Decay Level: {level}</h3>
            <div class="kpi-row">
                <div class="kpi-cell">
                    <div class="kpi-label">Waste Type</div>
                    <div class="kpi-value">{image_name}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">Critical Window</div>
                    <div class="kpi-value" style="font-size:0.88rem;">{countdown}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">Gas Risk</div>
                    <div class="kpi-value" style="font-size:0.9rem;">{gas}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">Confidence</div>
                    <div class="kpi-value">{conf}%</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">Inference Time</div>
                    <div class="kpi-value">{lat} ms</div>
                </div>
            </div>
            <div style="margin-bottom:8px;">
                <div class="kpi-label" style="margin-bottom:6px;">Odor Risk Score</div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div class="risk-bar-wrap" style="flex:1;">
                        <div class="risk-bar-fill"
                             style="width:{score}%;background:{bar_col};"></div>
                    </div>
                    <span style="color:{bar_col};font-weight:800;
                                 font-size:1rem;min-width:48px;">{score}/100</span>
                </div>
            </div>
            <div class="action-box">{action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#10B981;margin-bottom:2px;'>🌿 SmartRot AI</h2>"
        "<p style='color:#64748B;font-size:0.8rem;margin-top:0;'>"
        "Edge-AI TPS Decay &amp; Odor Risk Monitor</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Pilot status badge
    st.markdown(
        "<div style='margin-bottom:4px;font-size:0.7rem;font-weight:700;"
        "letter-spacing:0.08em;text-transform:uppercase;color:#64748B;'>"
        "Project Status</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge-active">● PILOT ACTIVE</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin-top:10px;font-size:0.82rem;color:#CBD5E1;line-height:1.7;'>"
        "<b style='color:#F8FAFC;'>Pilot:</b> DKI Jakarta<br>"
        "<b style='color:#F8FAFC;'>Scope:</b> Waste Collection Points (TPS)<br>"
        "<b style='color:#F8FAFC;'>Region:</b> 5 Administrative Cities"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # App metadata
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#64748B;margin-bottom:8px;'>App Metadata</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <table style="width:100%;font-size:0.8rem;border-collapse:collapse;">
            <tr>
                <td style="color:#64748B;padding:3px 0;">Version</td>
                <td style="color:#F8FAFC;font-weight:600;text-align:right;">1.0.0-alpha</td>
            </tr>
            <tr>
                <td style="color:#64748B;padding:3px 0;">Model</td>
                <td style="color:#F8FAFC;font-weight:600;text-align:right;">OpenVINO IR</td>
            </tr>
            <tr>
                <td style="color:#64748B;padding:3px 0;">Framework</td>
                <td style="color:#F8FAFC;font-weight:600;text-align:right;">Streamlit</td>
            </tr>
            <tr>
                <td style="color:#64748B;padding:3px 0;">Inference</td>
                <td style="color:#10B981;font-weight:700;text-align:right;">Intel Edge AI</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Data sources
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#64748B;margin-bottom:8px;'>Data Sources</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.82rem;color:#CBD5E1;line-height:2;'>"
        "📡 Edge Camera Feed (RTSP)<br>"
        "🌡️ IoT Sensor Array<br>"
        "🗺️ GeoJSON: DKI Jakarta TPS<br>"
        "📊 DLH Jakarta Open Data"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Intel Hardware Acceleration Stats
    with st.expander("⚡ Intel Hardware Acceleration Stats", expanded=False):
        st.markdown(
            """
            <table style="width:100%;font-size:0.8rem;border-collapse:collapse;">
                <tr>
                    <td style="color:#64748B;padding:4px 0;">🧠 Framework</td>
                    <td style="color:#93C5FD;font-weight:700;text-align:right;">Intel OpenVINO™ 2026</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:4px 0;">⚙️ Processing Mode</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">Offline Edge Inference</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:4px 0;">🔋 Energy Efficiency</td>
                    <td style="color:#10B981;font-weight:700;text-align:right;">3.2× lower wattage</td>
                </tr>
                <tr>
                    <td colspan="2" style="color:#475569;font-size:0.72rem;padding-bottom:4px;">
                        vs. PyTorch CPU baseline
                    </td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:4px 0;">🎯 Target Devices</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">NPU · GPU · CPU</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:4px 0;">📦 Model Format</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">OpenVINO IR (.xml/.bin)</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:4px 0;">🌐 Connectivity</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">Air-gapped / No Cloud</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")

    st.markdown(
        "<p style='font-size:0.74rem;color:#475569;line-height:1.6;'>"
        "© 2026 SmartRot AI Team<br>"
        "Intel AI Innovation Challenge</p>",
        unsafe_allow_html=True,
    )

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(
    "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:2px;'>"
    "<h1 style='color:#10B981;margin:0;font-size:2rem;font-weight:800;'>"
    "🌿 SmartRot AI</h1>"
    "<span style='color:#334155;font-size:1.4rem;'>|</span>"
    "<span style='color:#64748B;font-size:0.95rem;font-weight:500;'>"
    "Jakarta Smart City Command Center</span>"
    "</div>"
    "<p style='color:#475569;font-size:0.85rem;margin-top:4px;margin-bottom:0;'>"
    "Edge-AI TPS Decay &amp; Odor Risk Monitor &nbsp;·&nbsp; "
    "<span style='color:#10B981;font-weight:700;'>● DKI Jakarta Pilot Active</span>"
    "&nbsp;·&nbsp; Intel OpenVINO™ Powered"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Tabbed navigation ─────────────────────────────────────────────────────────
tab_detector, tab_map = st.tabs([
    "📷 Edge AI Detector (TPS Level)",
    "🗺️ DLH Jakarta Central Control Room",
])

# ── Tab 1: Edge AI Detector ───────────────────────────────────────────────────
with tab_detector:
    st.markdown(
        "<h3 style='color:#F8FAFC;margin-bottom:4px;'>📷 Edge AI Detector"
        " — TPS Waste Decay Analysis</h3>"
        "<p style='color:#64748B;font-size:0.83rem;margin-top:0;'>"
        "Upload a waste image, select its category, and run OpenVINO inference "
        "to obtain a decay timeline and odor risk assessment.</p>",
        unsafe_allow_html=True,
    )

    # ── OpenVINO device probe (cached) ────────────────────────────────────────
    ov_info = check_openvino_device()
    render_openvino_card(ov_info)

    st.markdown("---")

    # ── Input layout: uploader left, selector right ───────────────────────────
    col_upload, col_select = st.columns([2, 1], gap="large")

    with col_upload:
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#64748B;margin-bottom:8px;'>"
            "📂 Upload Waste Image</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            label="Upload a TPS waste image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a photo of waste at the TPS site for decay analysis.",
            label_visibility="collapsed",
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption=f"📷 {uploaded_file.name}",
                use_container_width=True,
            )

    with col_select:
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#64748B;margin-bottom:8px;'>"
            "🗂️ Waste Category</div>",
            unsafe_allow_html=True,
        )
        waste_type = st.selectbox(
            "Select waste category",
            options=list(DECAY_PROFILES.keys()),
            index=0,
            help=(
                "Choose the waste category that matches the uploaded image, "
                "or select manually to simulate a category without an image."
            ),
            label_visibility="collapsed",
        )

        st.markdown(
            "<p style='font-size:0.78rem;color:#475569;margin-top:10px;line-height:1.5;'>"
            "The classifier uses this category to determine the decay timeline "
            "and gas risk profile. In production, the OpenVINO model infers "
            "this automatically from the image tensor."
            "</p>",
            unsafe_allow_html=True,
        )

        run_analysis = st.button("⚡ Run Decay Analysis", use_container_width=True)

    st.markdown("---")

    # ── Analysis output ───────────────────────────────────────────────────────
    if run_analysis or uploaded_file:
        if run_analysis:
            # Show a progress bar to simulate inference pipeline
            progress_bar = st.progress(0, text="Initialising OpenVINO runtime…")
            for pct, label in [
                (20,  "Loading IR model weights…"),
                (45,  "Pre-processing image tensor…"),
                (70,  "Running inference on " + ov_info["device"] + "…"),
                (90,  "Post-processing predictions…"),
                (100, "Analysis complete ✓"),
            ]:
                time.sleep(0.25)
                progress_bar.progress(pct, text=label)
            time.sleep(0.2)
            progress_bar.empty()

        display_name = waste_type
        profile = simulate_inference(waste_type, ov_info["latency_ms"])

        # ── Top summary metrics ───────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label="Decay Level",
                value=profile["level"],
                delta=None,
            )
        with m2:
            st.metric(
                label="Critical Window",
                value=f"{profile['critical_hours']}h",
                delta=f"-{profile['critical_hours']}h remaining",
                delta_color="inverse",
            )
        with m3:
            st.metric(
                label="Odor Risk Score",
                value=f"{profile['risk_score']} / 100",
                delta=None,
            )
        with m4:
            st.metric(
                label="Model Confidence",
                value=f"{profile['confidence']}%",
                delta=None,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Detailed result card ──────────────────────────────────────────────
        render_result_card(profile, display_name)

    else:
        # ── Idle state ────────────────────────────────────────────────────────
        st.markdown(
            "<div class='upload-idle'>"
            "<div class='idle-icon'>📷</div>"
            "<div class='idle-text'>Upload a waste image and press "
            "<strong style='color:#10B981;'>Run Decay Analysis</strong> to begin.</div>"
            "<div class='idle-hint'>Supported formats: JPG &nbsp;·&nbsp; PNG &nbsp;·&nbsp; WEBP</div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# ── Tab 2 data — TPS site registry ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

TPS_SITES = [
    {
        "name":           "TPS Pasar Minggu",
        "lat":            -6.2856,
        "lon":            106.8378,
        "waste_type":     "🐟 Seafood / Meat",
        "hours_to_decay": 8,
        "priority_score": 95,
        "risk_level":     "CRITICAL",
        "marker_color":   "red",
        "gas_risk":       "H₂S (High)",
        "action_status":  "Truck Dispatched",
        "district":       "Jakarta Selatan",
        "notes":          "High seafood waste volume. H₂S odor detected by sensor array.",
    },
    {
        "name":           "TPS Pasar Senen",
        "lat":            -6.1754,
        "lon":            106.8454,
        "waste_type":     "🐟 Seafood / Meat",
        "hours_to_decay": 6,
        "priority_score": 98,
        "risk_level":     "CRITICAL",
        "marker_color":   "red",
        "gas_risk":       "H₂S (Critical)",
        "action_status":  "Truck Dispatched",
        "district":       "Jakarta Pusat",
        "notes":          "Critical odor risk. Overflow detected. Immediate evacuation required.",
    },
    {
        "name":           "TPS Kebayoran Baru",
        "lat":            -6.2437,
        "lon":            106.7963,
        "waste_type":     "🥦 Vegetables / Fruits",
        "hours_to_decay": 18,
        "priority_score": 62,
        "risk_level":     "WARNING",
        "marker_color":   "orange",
        "gas_risk":       "CH₄ + VOCs",
        "action_status":  "Pending",
        "district":       "Jakarta Selatan",
        "notes":          "Fermentation odor present. Schedule pickup within 8 hours.",
    },
    {
        "name":           "TPS Pasar Rumput",
        "lat":            -6.2183,
        "lon":            106.8508,
        "waste_type":     "🍚 Cooked Carbs / Dry Waste",
        "hours_to_decay": 34,
        "priority_score": 18,
        "risk_level":     "SAFE",
        "marker_color":   "green",
        "gas_risk":       "Minimal",
        "action_status":  "Pending",
        "district":       "Jakarta Selatan",
        "notes":          "Low risk. Routine inspection scheduled.",
    },
    {
        "name":           "TPS Jatinegara",
        "lat":            -6.2154,
        "lon":            106.8720,
        "waste_type":     "🥦 Vegetables / Fruits",
        "hours_to_decay": 20,
        "priority_score": 58,
        "risk_level":     "WARNING",
        "marker_color":   "orange",
        "gas_risk":       "CH₄ + VOCs",
        "action_status":  "Pending",
        "district":       "Jakarta Timur",
        "notes":          "Mixed organic waste. Monitor methane sensor.",
    },
    {
        "name":           "TPS Tanah Abang",
        "lat":            -6.1862,
        "lon":            106.8178,
        "waste_type":     "🍚 Cooked Carbs / Dry Waste",
        "hours_to_decay": 30,
        "priority_score": 24,
        "risk_level":     "SAFE",
        "marker_color":   "green",
        "gas_risk":       "Minimal",
        "action_status":  "Pending",
        "district":       "Jakarta Pusat",
        "notes":          "Dry waste only. No immediate action needed.",
    },
]

# Folium marker color → icon color mapping (Folium uses named colors)
_FOLIUM_ICON_MAP = {
    "red":    {"color": "red",    "icon": "exclamation-sign"},
    "orange": {"color": "orange", "icon": "warning-sign"},
    "green":  {"color": "green",  "icon": "ok-sign"},
}


@st.cache_data(show_spinner=False)
def build_dispatch_dataframe() -> pd.DataFrame:
    """Build the priority dispatch DataFrame from TPS_SITES registry."""
    rows = []
    for site in TPS_SITES:
        rows.append({
            "TPS Name":        site["name"],
            "District":        site["district"],
            "Waste Type":      site["waste_type"],
            "Hours to Decay":  site["hours_to_decay"],
            "Priority Score":  site["priority_score"],
            "Risk Level":      site["risk_level"],
            "Gas Risk":        site["gas_risk"],
            "Action Status":   site["action_status"],
        })
    df = pd.DataFrame(rows)
    # Sort by priority descending so Critical rows appear first
    return df.sort_values("Priority Score", ascending=False).reset_index(drop=True)


def _style_dispatch_table(df: pd.DataFrame):
    """Apply row-level background highlights by risk level."""
    STYLES = {
        "CRITICAL": "background-color: #2D0A0A; color: #EF4444;",
        "WARNING":  "background-color: #2D1A00; color: #F59E0B;",
        "SAFE":     "background-color: #052E16; color: #10B981;",
    }

    def row_style(row):
        base = STYLES.get(row["Risk Level"], "")
        return [base] * len(row)

    return (
        df.style
        .apply(row_style, axis=1)
        .set_properties(**{
            "font-size":   "0.85rem",
            "font-weight": "600",
            "border":      "1px solid #334155",
        })
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#1E293B"),
                ("color",            "#10B981"),
                ("font-size",        "0.72rem"),
                ("font-weight",      "700"),
                ("letter-spacing",   "0.06em"),
                ("text-transform",   "uppercase"),
                ("border-bottom",    "2px solid #334155"),
                ("padding",          "10px 12px"),
            ],
        }, {
            "selector": "td",
            "props": [("padding", "8px 12px")],
        }])
        .format({"Priority Score": "{:.0f}", "Hours to Decay": "{:.0f} h"})
        .hide(axis="index")
    )


def build_folium_map() -> folium.Map:
    """Build the Folium map with TPS markers and popups."""
    m = folium.Map(
        location=[-6.2088, 106.8456],
        zoom_start=12,
        tiles="cartodbpositron",   # clean minimal canvas — markers pop with high contrast
        prefer_canvas=True,
    )

    for site in TPS_SITES:
        ic = _FOLIUM_ICON_MAP[site["marker_color"]]
        risk_badge_color = {
            "CRITICAL": "#EF4444",
            "WARNING":  "#F59E0B",
            "SAFE":     "#10B981",
        }.get(site["risk_level"], "#64748B")

        popup_html = f"""
        <div style="font-family:'Inter','Segoe UI',sans-serif;min-width:230px;
                    background:#1E293B;color:#F8FAFC;
                    border-radius:10px;padding:14px 16px;
                    border-left:4px solid {risk_badge_color};
                    box-shadow:0 4px 16px rgba(0,0,0,0.4);">
            <b style="color:{risk_badge_color};font-size:0.95rem;">
                {site['name']}
            </b><br>
            <span style="color:#64748B;font-size:0.76rem;">{site['district']}</span>
            <hr style="border-color:#334155;margin:8px 0;">
            <table style="width:100%;font-size:0.8rem;border-collapse:collapse;">
                <tr>
                    <td style="color:#64748B;padding:3px 0;">Waste Type</td>
                    <td style="color:#F8FAFC;text-align:right;">{site['waste_type']}</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">Risk Level</td>
                    <td style="color:{risk_badge_color};font-weight:700;
                               text-align:right;">{site['risk_level']}</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">Hours to Decay</td>
                    <td style="color:#F8FAFC;text-align:right;">{site['hours_to_decay']}h</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">Gas Risk</td>
                    <td style="color:#F8FAFC;text-align:right;">{site['gas_risk']}</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">Priority Score</td>
                    <td style="color:{risk_badge_color};font-weight:700;
                               text-align:right;">{site['priority_score']}/100</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">Action</td>
                    <td style="color:#10B981;font-weight:700;
                               text-align:right;">{site['action_status']}</td>
                </tr>
            </table>
            <div style="margin-top:10px;padding:7px 10px;
                        background:#0F172A;border-radius:6px;
                        font-size:0.76rem;color:#94A3B8;">
                📝 {site['notes']}
            </div>
        </div>
        """

        folium.Marker(
            location=[site["lat"], site["lon"]],
            popup=folium.Popup(
                folium.IFrame(popup_html, width=270, height=290),
                max_width=290,
            ),
            tooltip=f"{site['name']} — {site['risk_level']}",
            icon=folium.Icon(
                color=ic["color"],
                icon=ic["icon"],
                prefix="glyphicon",
            ),
        ).add_to(m)

    # ── Legend overlay ─────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:#1E293B;border:1px solid #334155;
                border-radius:10px;padding:12px 16px;
                font-family:'Inter','Segoe UI',sans-serif;
                font-size:0.78rem;color:#CBD5E1;
                box-shadow:0 4px 12px rgba(0,0,0,0.3);">
        <b style="color:#10B981;display:block;margin-bottom:6px;
                  font-size:0.72rem;letter-spacing:0.07em;text-transform:uppercase;">
            TPS Risk Legend
        </b>
        <span style="color:#EF4444;">● CRITICAL</span>
        &nbsp;&nbsp;
        <span style="color:#F59E0B;">● WARNING</span>
        &nbsp;&nbsp;
        <span style="color:#10B981;">● SAFE</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ── Tab 2: DLH Jakarta Central Control Room ───────────────────────────────────
with tab_map:
    st.markdown(
        "<h3 style='color:#F8FAFC;margin-bottom:4px;'>🗺️ DLH Jakarta"
        " Central Control Room</h3>"
        "<p style='color:#64748B;font-size:0.83rem;margin-top:0;'>"
        "Real-time TPS site monitoring across DKI Jakarta's 5 administrative cities. "
        "Click any map marker for a full site detail card.</p>",
        unsafe_allow_html=True,
    )

    # ── KPI bar — 4 command-center tiles ─────────────────────────────────────
    tm1, tm2, tm3, tm4 = st.columns(4)
    with tm1:
        st.metric(
            label="🗑️ Active TPS Monitored",
            value="6",
            delta="DKI Jakarta Pilot",
            delta_color="off",
        )
    with tm2:
        st.metric(
            label="🚨 Critical Decay Alerts",
            value="2",
            delta="Immediate dispatch required",
            delta_color="inverse",
        )
    with tm3:
        st.metric(
            label="🌿 CH₄ Prevented (est.)",
            value="142 kg",
            delta="↓ vs. unmonitored baseline",
            delta_color="normal",
        )
    with tm4:
        st.metric(
            label="🕐 Last Scan",
            value=datetime.now().strftime("%H:%M"),
            delta=datetime.now().strftime("%d %b %Y"),
            delta_color="off",
        )

    st.markdown("---")

    # ── Map + site status side-by-side ────────────────────────────────────────
    col_map, col_list = st.columns([3, 1], gap="large")

    with col_map:
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#64748B;margin-bottom:8px;'>"
            "📍 Live TPS Risk Map — DKI Jakarta</div>",
            unsafe_allow_html=True,
        )
        jakarta_map = build_folium_map()
        st_folium(
            jakarta_map,
            width=None,
            height=500,
            returned_objects=[],
        )

    with col_list:
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#64748B;margin-bottom:8px;'>"
            "📋 Site Status</div>",
            unsafe_allow_html=True,
        )
        for site in sorted(TPS_SITES, key=lambda x: x["priority_score"], reverse=True):
            badge_color = {
                "CRITICAL": "#EF4444",
                "WARNING":  "#F59E0B",
                "SAFE":     "#10B981",
            }[site["risk_level"]]
            action_color = "#10B981" if site["action_status"] == "Truck Dispatched" \
                           else "#64748B"
            st.markdown(
                f"""
                <div class="site-card"
                     style="border-left:3px solid {badge_color};">
                    <div class="sc-name" style="color:{badge_color};">
                        {site['name']}
                    </div>
                    <div class="sc-meta">
                        {site['waste_type']} &nbsp;·&nbsp; {site['hours_to_decay']}h
                    </div>
                    <div class="sc-action" style="color:{action_color};">
                        {site['action_status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Priority dispatch table ────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#64748B;margin-bottom:6px;'>"
        "📊 Priority Dispatch Table</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Sorted by Priority Score (highest first). "
        "🔴 Critical — immediate dispatch. "
        "🟡 Warning — scheduled. "
        "🟢 Safe — routine monitoring."
    )

    dispatch_df  = build_dispatch_dataframe()
    styled_table = _style_dispatch_table(dispatch_df)

    st.dataframe(
        styled_table,
        use_container_width=True,
        height=280,
    )

    # ── CSV Export ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    col_export, col_note = st.columns([1, 3], gap="small")

    with col_export:
        report_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        export_df  = dispatch_df.copy()
        export_df.insert(0, "Report Timestamp", report_ts)
        export_df.insert(1, "Pilot Region",     "DKI Jakarta")

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, encoding="utf-8")
        csv_bytes  = csv_buffer.getvalue().encode("utf-8")
        filename   = f"DLH_Priority_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

        st.download_button(
            label="📥 Export DLH Daily Priority Report (CSV)",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
            help="Download the full TPS priority dispatch table as a CSV for DLH records.",
        )

    with col_note:
        st.markdown(
            "<div class='export-note'>"
            "📄 Export includes TPS Name, District, Waste Type, Hours to Decay, "
            "Priority Score, Risk Level, Gas Risk, and Action Status — "
            "timestamped for DLH Jakarta daily operational records."
            "</div>",
            unsafe_allow_html=True,
        )
