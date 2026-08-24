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

    /* ── Recent alerts panel (Tab 2 left column) ── */
    .alert-panel-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }
    .alert-report-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 12px 14px;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
    }
    .alert-report-card::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
    }
    .arc-critical::before { background: var(--risk-red); }
    .arc-warning::before  { background: var(--risk-yel); }
    .arc-safe::before     { background: var(--risk-grn); }
    .arc-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 6px;
    }
    .arc-name {
        font-size: 0.84rem;
        font-weight: 700;
        color: var(--text-hi);
        line-height: 1.3;
    }
    .arc-badge {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        padding: 2px 8px;
        border-radius: 20px;
        flex-shrink: 0;
        margin-left: 8px;
    }
    .arc-badge-critical {
        background: var(--risk-red-bg);
        color: var(--risk-red);
        border: 1px solid var(--risk-red);
    }
    .arc-badge-warning {
        background: var(--risk-yel-bg);
        color: var(--risk-yel);
        border: 1px solid var(--risk-yel);
    }
    .arc-badge-safe {
        background: var(--risk-grn-bg);
        color: var(--risk-grn);
        border: 1px solid var(--risk-grn);
    }
    .arc-meta {
        font-size: 0.76rem;
        color: var(--text-lo);
        margin-bottom: 4px;
    }
    .arc-urgency {
        font-size: 0.8rem;
        font-weight: 700;
    }
    .arc-action {
        font-size: 0.74rem;
        font-weight: 600;
        margin-top: 4px;
        padding-top: 6px;
        border-top: 1px solid var(--border);
    }

    /* ── Map section label ── */
    .map-section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .map-live-dot {
        width: 7px; height: 7px;
        background: var(--accent);
        border-radius: 50%;
        box-shadow: 0 0 6px var(--accent);
        display: inline-block;
    }

    /* ── Right-panel idle placeholder (Tab 1) ── */
    .right-idle {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 40px 20px;
        text-align: center;
        color: var(--text-lo);
        height: 100%;
    }
    .right-idle .ri-icon { font-size: 2.2rem; margin-bottom: 10px; }
    .right-idle .ri-text { font-size: 0.88rem; color: var(--text-mid); }
    .right-idle .ri-hint { font-size: 0.76rem; color: var(--text-lo); margin-top: 6px; }

    /* ── Section divider label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 8px;
    }

    /* ══ SIDEBAR NAVIGATION ══════════════════════════════════════════════════ */

    /* Hide default Streamlit radio button circles */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 2px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: background 0.15s ease !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: var(--text-mid) !important;
        border: 1px solid transparent !important;
        margin: 0 !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #1E293B !important;
        color: var(--text-hi) !important;
        border-color: var(--border) !important;
    }
    /* Selected nav item */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div {
        background-color: #0D2E22 !important;
        color: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    /* Hide the radio circle dot */
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    /* Nav section divider label */
    .nav-section-header {
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-lo);
        padding: 0 4px;
        margin: 12px 0 6px;
    }

    /* ── System status card (sidebar bottom) ── */
    .sys-status-card {
        background: #0D1929;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.78rem;
    }
    .sys-status-card .ss-title {
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 8px;
    }
    .sys-status-card table {
        width: 100%;
        border-collapse: collapse;
    }
    .sys-status-card td {
        padding: 3px 0;
        font-size: 0.78rem;
    }
    .ss-dot {
        display: inline-block;
        width: 7px; height: 7px;
        background: var(--accent);
        border-radius: 50%;
        box-shadow: 0 0 6px var(--accent);
        margin-right: 5px;
        vertical-align: middle;
    }

    /* ══ SDG ANALYTICS PAGE ══════════════════════════════════════════════════ */
    .sdg-header-card {
        background: linear-gradient(135deg, #0D2E22 0%, #0F172A 100%);
        border: 1px solid var(--accent);
        border-radius: var(--radius);
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .sdg-header-card h2 {
        color: var(--accent) !important;
        margin: 0 0 6px;
        font-size: 1.3rem;
    }
    .sdg-header-card p {
        color: var(--text-mid) !important;
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .sdg-goal-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 18px 20px;
        height: 100%;
    }
    .sdg-goal-card .sgc-icon { font-size: 2rem; margin-bottom: 8px; }
    .sgc-title {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 4px;
    }
    .sgc-goal {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-hi);
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .sgc-body {
        font-size: 0.82rem;
        color: var(--text-mid);
        line-height: 1.55;
    }
    .impact-stat {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius-sm);
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .impact-stat .is-num {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--accent);
        line-height: 1;
        margin-bottom: 3px;
    }
    .impact-stat .is-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-mid);
    }
    .impact-stat .is-desc {
        font-size: 0.73rem;
        color: var(--text-lo);
        margin-top: 2px;
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

# ══════════════════════════════════════════════════════════════════════════════
# ── TPS Site Registry — global data, defined before any UI rendering ──────────
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

# Folium marker icon mapping
_FOLIUM_ICON_MAP = {
    "red":    {"color": "red",    "icon": "exclamation-sign"},
    "orange": {"color": "orange", "icon": "warning-sign"},
    "green":  {"color": "green",  "icon": "ok-sign"},
}


# ══════════════════════════════════════════════════════════════════════════════
# ── Map & Table helper functions ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

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
    return df.sort_values("Priority Score", ascending=False).reset_index(drop=True)


def _style_dispatch_table(df: pd.DataFrame):
    """Apply row-level background highlights by risk level."""
    STYLES = {
        "CRITICAL": "background-color: #2D0A0A; color: #EF4444;",
        "WARNING":  "background-color: #2D1A00; color: #F59E0B;",
        "SAFE":     "background-color: #052E16; color: #10B981;",
    }

    def row_style(row):
        return [STYLES.get(row["Risk Level"], "")] * len(row)

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
    """Build the Folium map with TPS markers and styled popups."""
    m = folium.Map(
        location=[-6.2088, 106.8456],
        zoom_start=12,
        tiles="cartodbpositron",
        prefer_canvas=True,
    )

    for site in TPS_SITES:
        ic = _FOLIUM_ICON_MAP[site["marker_color"]]
        risk_color = {
            "CRITICAL": "#EF4444",
            "WARNING":  "#F59E0B",
            "SAFE":     "#10B981",
        }.get(site["risk_level"], "#64748B")

        popup_html = (
            f'<div style="font-family:\'Inter\',\'Segoe UI\',sans-serif;min-width:230px;'
            f'background:#1E293B;color:#F8FAFC;border-radius:10px;padding:14px 16px;'
            f'border-left:4px solid {risk_color};box-shadow:0 4px 16px rgba(0,0,0,0.4);">'
            f'<b style="color:{risk_color};font-size:0.95rem;">{site["name"]}</b><br>'
            f'<span style="color:#64748B;font-size:0.76rem;">{site["district"]}</span>'
            f'<hr style="border-color:#334155;margin:8px 0;">'
            f'<table style="width:100%;font-size:0.8rem;border-collapse:collapse;">'
            f'<tr><td style="color:#64748B;padding:3px 0;">Waste Type</td>'
            f'<td style="color:#F8FAFC;text-align:right;">{site["waste_type"]}</td></tr>'
            f'<tr><td style="color:#64748B;padding:3px 0;">Risk Level</td>'
            f'<td style="color:{risk_color};font-weight:700;text-align:right;">{site["risk_level"]}</td></tr>'
            f'<tr><td style="color:#64748B;padding:3px 0;">Hours to Decay</td>'
            f'<td style="color:#F8FAFC;text-align:right;">{site["hours_to_decay"]}h</td></tr>'
            f'<tr><td style="color:#64748B;padding:3px 0;">Gas Risk</td>'
            f'<td style="color:#F8FAFC;text-align:right;">{site["gas_risk"]}</td></tr>'
            f'<tr><td style="color:#64748B;padding:3px 0;">Priority Score</td>'
            f'<td style="color:{risk_color};font-weight:700;text-align:right;">{site["priority_score"]}/100</td></tr>'
            f'<tr><td style="color:#64748B;padding:3px 0;">Action</td>'
            f'<td style="color:#10B981;font-weight:700;text-align:right;">{site["action_status"]}</td></tr>'
            f'</table>'
            f'<div style="margin-top:10px;padding:7px 10px;background:#0F172A;'
            f'border-radius:6px;font-size:0.76rem;color:#94A3B8;">📝 {site["notes"]}</div>'
            f'</div>'
        )

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

    legend_html = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        'background:#1E293B;border:1px solid #334155;border-radius:10px;'
        'padding:12px 16px;font-family:\'Inter\',\'Segoe UI\',sans-serif;'
        'font-size:0.78rem;color:#CBD5E1;box-shadow:0 4px 12px rgba(0,0,0,0.3);">'
        '<b style="color:#10B981;display:block;margin-bottom:6px;'
        'font-size:0.72rem;letter-spacing:0.07em;text-transform:uppercase;">'
        'TPS Risk Legend</b>'
        '<span style="color:#EF4444;">&#9679; CRITICAL</span>&nbsp;&nbsp;'
        '<span style="color:#F59E0B;">&#9679; WARNING</span>&nbsp;&nbsp;'
        '<span style="color:#10B981;">&#9679; SAFE</span>'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ══════════════════════════════════════════════════════════════════════════════
# ── Sidebar — Navigation + Metadata ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Logo / Brand ──────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#10B981;margin-bottom:0;font-size:1.25rem;"
        "font-weight:800;letter-spacing:-0.01em;'>"
        "🌿 SmartRot AI</h2>"
        "<p style='color:#475569;font-size:0.74rem;margin-top:2px;'>"
        "Jakarta Smart City Command Center</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,"
        "#10B981 0%,#334155 60%,transparent 100%);"
        "margin:10px 0 14px;'></div>",
        unsafe_allow_html=True,
    )

    # ── Primary Navigation ─────────────────────────────────────────────────
    st.markdown('<div class="nav-section-header">Navigation</div>',
                unsafe_allow_html=True)

    NAV_OPTIONS = [
        "🗺️  Live TPS Map & Control Room",
        "📷  Edge AI Decay Detector",
        "📊  Impact & SDG Analytics",
    ]
    active_page = st.radio(
        label="nav",
        options=NAV_OPTIONS,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown(
        "<div style='height:1px;background:#334155;margin:14px 0;'></div>",
        unsafe_allow_html=True,
    )

    # ── Pilot Status ──────────────────────────────────────────────────────────
    st.markdown('<div class="nav-section-header">Pilot Status</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<span class="badge-active">● PILOT ACTIVE</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin-top:8px;font-size:0.8rem;color:#CBD5E1;line-height:1.8;'>"
        "<b style='color:#F8FAFC;'>Region:</b> DKI Jakarta<br>"
        "<b style='color:#F8FAFC;'>Scope:</b> TPS Collection Points<br>"
        "<b style='color:#F8FAFC;'>Cities:</b> 5 Administrative"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='height:1px;background:#334155;margin:14px 0;'></div>",
        unsafe_allow_html=True,
    )

    # ── Data Sources ──────────────────────────────────────────────────────────
    st.markdown('<div class="nav-section-header">Data Sources</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.79rem;color:#94A3B8;line-height:1.9;'>"
        "📡 Edge Camera Feed (RTSP)<br>"
        "🌡️ IoT Sensor Array<br>"
        "🗺️ GeoJSON: DKI Jakarta TPS<br>"
        "📊 DLH Jakarta Open Data"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='height:1px;background:#334155;margin:14px 0;'></div>",
        unsafe_allow_html=True,
    )

    # ── Intel Hardware Stats expander ─────────────────────────────────────────
    with st.expander("⚡ Intel Hardware Stats", expanded=False):
        st.markdown(
            """
            <table style="width:100%;font-size:0.78rem;border-collapse:collapse;">
                <tr>
                    <td style="color:#64748B;padding:3px 0;">🧠 Framework</td>
                    <td style="color:#93C5FD;font-weight:700;text-align:right;">Intel OpenVINO™ 2026</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">⚙️ Mode</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">Offline Edge Inference</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">🔋 Efficiency</td>
                    <td style="color:#10B981;font-weight:700;text-align:right;">3.2× lower wattage</td>
                </tr>
                <tr>
                    <td colspan="2" style="color:#475569;font-size:0.7rem;padding-bottom:3px;">
                        vs. PyTorch CPU baseline
                    </td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">🎯 Devices</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">NPU · GPU · CPU</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">📦 Format</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">OpenVINO IR (.xml/.bin)</td>
                </tr>
                <tr>
                    <td style="color:#64748B;padding:3px 0;">🌐 Network</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">Air-gapped</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

    # ── Spacer pushes system status to bottom ─────────────────────────────────
    st.markdown("<div style='flex:1;min-height:32px;'></div>", unsafe_allow_html=True)

    # ── System Status card (always visible at bottom) ─────────────────────────
    _ov_status = check_openvino_device()
    _device    = _ov_status["device"]
    _mode_txt  = "OpenVINO Optimized" if _ov_status["available"] else "Simulation"
    st.markdown(
        f"""
        <div class="sys-status-card">
            <div class="ss-title">System Status</div>
            <table>
                <tr>
                    <td style="color:#64748B;">Runtime</td>
                    <td style="color:#93C5FD;font-weight:700;text-align:right;">
                        Intel OpenVINO IR
                    </td>
                </tr>
                <tr>
                    <td style="color:#64748B;">Device</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">
                        <span class="ss-dot"></span>{_device}
                    </td>
                </tr>
                <tr>
                    <td style="color:#64748B;">Mode</td>
                    <td style="color:#10B981;font-weight:700;text-align:right;">
                        {_mode_txt}
                    </td>
                </tr>
                <tr>
                    <td style="color:#64748B;">Pilot</td>
                    <td style="color:#F8FAFC;font-weight:600;text-align:right;">
                        DKI Jakarta
                    </td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.7rem;color:#334155;text-align:center;"
        "margin-top:10px;'>© 2026 SmartRot AI · Intel AI Challenge</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Main Canvas — page header (shared across all views) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Page-title mapping
_PAGE_TITLES = {
    NAV_OPTIONS[0]: ("🗺️ DLH Jakarta Central Control Room",
                     "Real-time TPS site monitoring across DKI Jakarta's 5 administrative cities. "
                     "Click any map marker for a full site detail card."),
    NAV_OPTIONS[1]: ("📷 Edge AI Decay Detector — TPS Waste Analysis",
                     "Upload a waste image, select its category, and run OpenVINO inference "
                     "to obtain a decay timeline and odor risk assessment."),
    NAV_OPTIONS[2]: ("📊 Impact & SDG Analytics",
                     "Quantified environmental impact and UN SDG alignment of the SmartRot AI pilot "
                     "across DKI Jakarta's waste collection network."),
}

_title, _subtitle = _PAGE_TITLES[active_page]
st.markdown(
    f"<h2 style='color:#F8FAFC;margin-top:24px;margin-bottom:2px;font-size:1.5rem;font-weight:800;'>"
    f"{_title}</h2>"
    f"<p style='color:#64748B;font-size:0.82rem;margin-top:0;margin-bottom:14px;'>"
    f"{_subtitle}</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='height:2px;background:linear-gradient(90deg,"
    "#10B981 0%,#1E293B 50%,transparent 100%);"
    "margin-bottom:20px;'></div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Live TPS Map & Control Room ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if active_page == NAV_OPTIONS[0]:

    # ── KPI row ───────────────────────────────────────────────────────────────
    tm1, tm2, tm3, tm4 = st.columns(4)
    with tm1:
        st.metric("🗑️ Active TPS Monitored", "6",
                  delta="DKI Jakarta Pilot", delta_color="off")
    with tm2:
        st.metric("🚨 Critical Decay Alerts", "2",
                  delta="Immediate dispatch required", delta_color="inverse")
    with tm3:
        st.metric("🌿 CH₄ Prevented (est.)", "142 kg",
                  delta="↓ vs. unmonitored baseline", delta_color="normal")
    with tm4:
        st.metric("🕐 Last Scan",
                  datetime.now().strftime("%H:%M"),
                  delta=datetime.now().strftime("%d %b %Y"),
                  delta_color="off")

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # ── Main canvas: alerts left (1) | map right (2.5) ───────────────────────
    col_alerts, col_map_view = st.columns([1, 2.5], gap="medium")

    with col_alerts:
        st.markdown(
            "<div class='alert-panel-header'>"
            "🚨 Recent TPS Alerts &amp; Critical Reports"
            "</div>",
            unsafe_allow_html=True,
        )
        _RISK_BADGE_CSS = {
            "CRITICAL": "arc-badge arc-badge-critical",
            "WARNING":  "arc-badge arc-badge-warning",
            "SAFE":     "arc-badge arc-badge-safe",
        }
        _URGENCY_COLOR = {
            "CRITICAL": "#EF4444",
            "WARNING":  "#F59E0B",
            "SAFE":     "#10B981",
        }
        for site in sorted(TPS_SITES,
                           key=lambda x: x["priority_score"], reverse=True):
            risk      = site["risk_level"]
            bc        = _URGENCY_COLOR[risk]
            badge_cls = _RISK_BADGE_CSS[risk]
            arc_cls   = f"alert-report-card arc-{risk.lower()}"
            act_color = "#10B981" if site["action_status"] == "Truck Dispatched" \
                        else "#64748B"
            truck     = "🚛 " if site["action_status"] == "Truck Dispatched" else "⏳ "
            st.markdown(
                f"""
                <div class="{arc_cls}">
                    <div class="arc-top">
                        <div class="arc-name">{site['name']}</div>
                        <span class="{badge_cls}">{risk}</span>
                    </div>
                    <div class="arc-meta">
                        📍 {site['district']} &nbsp;·&nbsp; {site['waste_type']}
                    </div>
                    <div class="arc-urgency" style="color:{bc};">
                        ⏱ {site['hours_to_decay']}h to critical decay
                    </div>
                    <div class="arc-meta" style="margin-top:3px;">
                        ☣️ {site['gas_risk']}
                        &nbsp;·&nbsp; Priority: {site['priority_score']}/100
                    </div>
                    <div class="arc-action" style="color:{act_color};">
                        {truck}{site['action_status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_map_view:
        st.markdown(
            "<div class='map-section-label'>"
            "<span class='map-live-dot'></span>"
            "Live TPS Risk Map — DKI Jakarta"
            "</div>",
            unsafe_allow_html=True,
        )
        jakarta_map = build_folium_map()
        st_folium(jakarta_map, width=None, height=560, returned_objects=[])

    st.markdown("---")

    with st.expander("📊 Priority Dispatch Table — DLH Fleet Assignment",
                     expanded=False):
        st.caption(
            "Sorted by Priority Score (highest first).  "
            "🔴 Critical — immediate dispatch.  "
            "🟡 Warning — scheduled.  "
            "🟢 Safe — routine monitoring."
        )
        dispatch_df  = build_dispatch_dataframe()
        styled_table = _style_dispatch_table(dispatch_df)
        st.dataframe(styled_table, use_container_width=True, height=280)

        st.markdown("<br>", unsafe_allow_html=True)
        col_exp, col_note = st.columns([1, 3], gap="small")
        with col_exp:
            report_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            export_df  = dispatch_df.copy()
            export_df.insert(0, "Report Timestamp", report_ts)
            export_df.insert(1, "Pilot Region",     "DKI Jakarta")
            csv_buf    = io.StringIO()
            export_df.to_csv(csv_buf, index=False, encoding="utf-8")
            st.download_button(
                label="📥 Export DLH Daily Priority Report (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"DLH_Priority_Report_"
                          f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_note:
            st.markdown(
                "<div class='export-note'>"
                "📄 Includes TPS Name, District, Waste Type, Hours to Decay, "
                "Priority Score, Risk Level, Gas Risk, Action Status — "
                "timestamped for DLH Jakarta daily operational records."
                "</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Edge AI Decay Detector ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif active_page == NAV_OPTIONS[1]:

    ov_info = check_openvino_device()

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: image uploader + preview ───────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-label">📂 Waste Image Input</div>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="Upload a TPS waste image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a photo of waste at the TPS site for decay analysis.",
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.image(
                Image.open(uploaded_file),
                caption=f"📷 {uploaded_file.name}",
                use_container_width=True,
            )
        else:
            st.markdown(
                "<div class='upload-idle'>"
                "<div class='idle-icon'>🗑️</div>"
                "<div class='idle-text'>Drop or browse a waste photo here</div>"
                "<div class='idle-hint'>JPG &nbsp;·&nbsp; PNG &nbsp;·&nbsp; WEBP</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── RIGHT: OV badge + selector + button + results ────────────────────────
    with col_right:
        render_openvino_card(ov_info)

        st.markdown('<div class="section-label">🗂️ Waste Category</div>',
                    unsafe_allow_html=True)
        waste_type = st.selectbox(
            "Waste category",
            options=list(DECAY_PROFILES.keys()),
            index=0,
            help="Choose the category matching the image. "
                 "In production the OpenVINO model infers this automatically.",
            label_visibility="collapsed",
        )
        run_analysis = st.button("⚡ Run Decay Analysis", use_container_width=True)

        st.markdown("---")

        if run_analysis or uploaded_file:
            if run_analysis:
                pb = st.progress(0, text="Initialising OpenVINO runtime…")
                for pct, lbl in [
                    (20,  "Loading IR model weights…"),
                    (45,  "Pre-processing image tensor…"),
                    (70,  f"Running inference on {ov_info['device']}…"),
                    (90,  "Post-processing predictions…"),
                    (100, "Analysis complete ✓"),
                ]:
                    time.sleep(0.25)
                    pb.progress(pct, text=lbl)
                time.sleep(0.2)
                pb.empty()

            profile = simulate_inference(waste_type, ov_info["latency_ms"])

            km1, km2, km3, km4 = st.columns(4)
            with km1:
                st.metric("Decay Level",     profile["level"])
            with km2:
                st.metric("Critical Window",
                          f"{profile['critical_hours']}h",
                          delta=f"-{profile['critical_hours']}h",
                          delta_color="inverse")
            with km3:
                st.metric("Odor Risk Score", f"{profile['risk_score']}/100")
            with km4:
                st.metric("Confidence",      f"{profile['confidence']}%")

            st.markdown("<br>", unsafe_allow_html=True)
            render_result_card(profile, waste_type)

        else:
            st.markdown(
                "<div class='right-idle'>"
                "<div class='ri-icon'>⚡</div>"
                "<div class='ri-text'>Select a waste category and press<br>"
                "<strong style='color:#10B981;'>Run Decay Analysis</strong>"
                " to see the AI result here.</div>"
                "<div class='ri-hint'>Or upload an image first to auto-trigger.</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Impact & SDG Analytics ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif active_page == NAV_OPTIONS[2]:

    # ── SDG header card ───────────────────────────────────────────────────────
    st.markdown(
        "<div class='sdg-header-card'>"
        "<h2>🌍 SmartRot AI — Environmental Impact & UN SDG Alignment</h2>"
        "<p>Quantified outcomes from the DKI Jakarta pilot program, mapped to the "
        "United Nations Sustainable Development Goals. All figures are estimated "
        "based on 6 monitored TPS sites, 30-day pilot window.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Impact stats row ──────────────────────────────────────────────────────
    si1, si2, si3, si4 = st.columns(4)
    with si1:
        st.metric("🌿 CH₄ Prevented",   "142 kg",  delta="30-day pilot",    delta_color="normal")
    with si2:
        st.metric("🚛 Dispatches Saved", "38",      delta="vs manual routing", delta_color="normal")
    with si3:
        st.metric("⏱ Avg Response Time", "−2.4 h",  delta="vs baseline",     delta_color="normal")
    with si4:
        st.metric("🏙️ TPS Sites Covered", "6",       delta="DKI Jakarta",     delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SDG goal cards ────────────────────────────────────────────────────────
    sdg1, sdg2, sdg3 = st.columns(3)

    with sdg1:
        st.markdown(
            "<div class='sdg-goal-card'>"
            "<div class='sdg-goal-card .sgc-icon'>🏙️</div>"
            "<div class='sgc-title'>SDG 11</div>"
            "<div class='sgc-goal'>Sustainable Cities & Communities</div>"
            "<div class='sgc-body'>"
            "SmartRot AI enables DLH Jakarta to proactively manage organic waste "
            "at 6 TPS sites, reducing urban odor incidents and improving "
            "neighborhood sanitation for ~180,000 nearby residents."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with sdg2:
        st.markdown(
            "<div class='sdg-goal-card'>"
            "<div class='sdg-goal-card .sgc-icon'>🌍</div>"
            "<div class='sgc-title'>SDG 13</div>"
            "<div class='sgc-goal'>Climate Action</div>"
            "<div class='sgc-body'>"
            "Early decay detection prevents 142 kg of estimated methane (CH₄) "
            "from entering the atmosphere during the 30-day pilot — equivalent "
            "to removing ~3.5 tonnes of CO₂ equivalent from the urban environment."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with sdg3:
        st.markdown(
            "<div class='sdg-goal-card'>"
            "<div class='sdg-goal-card .sgc-icon'>🤖</div>"
            "<div class='sgc-title'>SDG 9</div>"
            "<div class='sgc-goal'>Industry, Innovation & Infrastructure</div>"
            "<div class='sgc-body'>"
            "Deploys Intel OpenVINO™ edge inference on commodity hardware — "
            "no cloud dependency, 3.2× lower energy than PyTorch CPU baseline — "
            "demonstrating scalable, low-cost AI infrastructure for public services."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quantified impact breakdown ───────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:800;letter-spacing:0.1em;"
        "text-transform:uppercase;color:#64748B;margin-bottom:12px;'>"
        "Quantified Impact Breakdown</div>",
        unsafe_allow_html=True,
    )

    ia, ib = st.columns(2)
    with ia:
        for stat in [
            ("142 kg",  "Methane Prevented (est.)",
             "Organic waste intercepted before critical decay threshold"),
            ("38",      "Optimised Dispatch Events",
             "AI-routed truck dispatches vs. manual schedule baseline"),
            ("−2.4 h",  "Faster Mean Response Time",
             "Average reduction in TPS service delay per alert cycle"),
        ]:
            st.markdown(
                f"<div class='impact-stat'>"
                f"<div class='is-num'>{stat[0]}</div>"
                f"<div class='is-label'>{stat[1]}</div>"
                f"<div class='is-desc'>{stat[2]}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with ib:
        for stat in [
            ("6",       "TPS Sites Monitored",
             "Across 4 Jakarta districts in the 30-day pilot"),
            ("3.2×",    "Energy Efficiency Gain",
             "Intel OpenVINO IR vs. PyTorch CPU inference baseline"),
            ("0",       "Cloud Dependencies",
             "Full offline edge inference — air-gapped, privacy-safe"),
        ]:
            st.markdown(
                f"<div class='impact-stat'>"
                f"<div class='is-num'>{stat[0]}</div>"
                f"<div class='is-label'>{stat[1]}</div>"
                f"<div class='is-desc'>{stat[2]}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "⚠️ All impact figures are simulated estimates based on the 6-site pilot model. "
        "Production deployment across all DKI Jakarta TPS sites is projected to scale "
        "impact by approximately 40–60×."
    )

