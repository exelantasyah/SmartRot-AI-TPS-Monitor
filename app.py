"""
SmartRot AI: Edge-AI TPS Decay & Odor Risk Monitor (DKI Jakarta)
Pilot Project — DKI Jakarta
"""

import io
import random
import textwrap
import time
from datetime import datetime, timedelta

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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

# ── Professional Operational UI — DLH Jakarta Control Room ────────────────────
SMARTCITY_CSS = """
<style>
    /* ══ DESIGN TOKENS ════════════════════════════════════════════════════════ */
    :root {
        /* Surfaces */
        --bg-app:      #0B1220;
        --bg-surface:  #111A2B;
        --bg-surface2: #182338;
        --bg-raised:   #1C2A3A;

        /* Borders */
        --border:      #26344A;
        --border-mid:  #2F4060;

        /* Text */
        --text-hi:     #E7EDF5;
        --text-mid:    #8D9AAF;
        --text-lo:     #4A5568;

        /* Semantic colors */
        --green:       #18B981;   /* operational / safe / healthy */
        --red:         #EF4444;   /* critical */
        --amber:       #F59E0B;   /* warning */
        --blue:        #3B82F6;   /* informational */

        /* Semantic backgrounds */
        --red-bg:      #1C0A0A;
        --amber-bg:    #1C1000;
        --green-bg:    #061A10;
        --blue-bg:     #08152A;

        /* Geometry */
        --radius:      6px;
        --radius-sm:   4px;
        --radius-md:   8px;
    }

    /* ══ APP SHELL ════════════════════════════════════════════════════════════ */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background-color: var(--bg-app) !important;
        color: var(--text-hi) !important;
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    }

    /* ══ STREAMLIT HEADER BAR ═════════════════════════════════════════════════ */
    [data-testid="stHeader"] {
        background-color: var(--bg-app) !important;
        border-bottom: 1px solid var(--border) !important;
    }

    /* ══ SIDEBAR ══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #0D1625 !important;
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
        color: var(--text-hi) !important;
    }

    /* ══ MAIN CONTENT CONTAINER ═══════════════════════════════════════════════ */
    [data-testid="stMainBlockContainer"] {
        background-color: var(--bg-app) !important;
        padding-top: 1.5rem;
    }
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

    /* ══ TABS ══════════════════════════════════════════════════════════════════ */
    [data-testid="stTabs"] [role="tablist"] {
        background-color: var(--bg-surface);
        border-radius: var(--radius) var(--radius) 0 0;
        padding: 6px 10px 0;
        border-bottom: 1px solid var(--border);
        gap: 4px;
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: var(--text-lo) !important;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 8px 18px;
        border-radius: var(--radius) var(--radius) 0 0;
        border: 1px solid transparent;
        transition: color 0.15s ease, background-color 0.15s ease;
        background: transparent;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--text-hi) !important;
        background-color: var(--bg-surface2);
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--text-hi) !important;
        background-color: var(--bg-app) !important;
        border-color: var(--border) var(--border) transparent !important;
        border-bottom: 2px solid var(--green) !important;
    }

    /* ══ STREAMLIT METRICS ════════════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 16px 18px !important;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"] {
        color: var(--text-lo) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-hi) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        color: var(--text-lo) !important;
        font-size: 0.75rem !important;
    }

    /* ══ BUTTONS ══════════════════════════════════════════════════════════════ */
    /* Base — all stButton and stDownloadButton instances                        */
    .stButton > button,
    .stDownloadButton > button,
    div.stButton > button,
    div.stButton > button[kind="primary"],
    div.stButton > button[kind="secondary"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.25rem !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    /* Hover */
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div.stButton > button:hover {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.5) !important;
        cursor: pointer !important;
    }
    /* Disabled — stays green, dimmed opacity */
    .stButton > button:disabled,
    .stButton > button[disabled],
    div.stButton > button:disabled,
    div.stButton > button[disabled] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        opacity: 0.45 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }
    /* Force inner span/p text colour (Streamlit injects a <p> inside button) */
    div.stButton > button p,
    div.stButton > button span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* ══ SELECTBOX / FILE UPLOADER ════════════════════════════════════════════ */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stFileUploader"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-hi) !important;
    }
    [data-testid="stFileUploader"] label {
        color: var(--text-mid) !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: var(--bg-surface2) !important;
        border: 1px dashed var(--border-mid) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] span {
        color: var(--text-mid) !important;
    }

    /* ══ PROGRESS BAR ═════════════════════════════════════════════════════════ */
    [data-testid="stProgressBar"] > div > div {
        background-color: var(--green) !important;
        border-radius: 3px;
    }
    [data-testid="stProgressBar"] > div {
        background-color: var(--border) !important;
        border-radius: 3px;
    }

    /* ══ EXPANDER ═════════════════════════════════════════════════════════════ */
    [data-testid="stExpander"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-hi) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    /* ══ DATAFRAME ════════════════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius) !important;
        overflow: hidden;
        border: 1px solid var(--border) !important;
    }

    /* ══ CAPTION / SMALL TEXT ═════════════════════════════════════════════════ */
    [data-testid="stCaptionContainer"] p,
    small {
        color: var(--text-lo) !important;
    }

    /* ══ INFO / ALERT BOX ═════════════════════════════════════════════════════ */
    [data-testid="stAlert"] {
        background-color: var(--bg-surface) !important;
        border-color: var(--border) !important;
        color: var(--text-mid) !important;
        border-radius: var(--radius) !important;
    }

    /* ══ DIVIDER ══════════════════════════════════════════════════════════════ */
    hr { border-color: var(--border) !important; }

    /* ══ SCROLLBAR ════════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-app); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-mid); }

    /* ══ SIDEBAR NAVIGATION ═══════════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 2px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 9px 12px !important;
        border-radius: var(--radius) !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--text-mid) !important;
        border: 1px solid transparent !important;
        margin: 0 !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: var(--bg-surface2) !important;
        color: var(--text-hi) !important;
    }
    /* Active nav item — subtle, not neon */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div {
        background-color: #0E1E30 !important;
        color: var(--text-hi) !important;
        border-color: var(--border-mid) !important;
        border-left: 2px solid var(--green) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }

    /* ── Sidebar section label ── */
    .sb-section {
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-lo);
        padding: 0 2px;
        margin: 16px 0 6px;
    }

    /* ── Sidebar rule ── */
    .sb-rule {
        height: 1px;
        background: var(--border);
        margin: 12px 0;
    }

    /* ── Sidebar system status block ── */
    .sys-block {
        background: #090F1C;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 12px 14px;
        font-size: 0.78rem;
    }
    .sys-block .sb-label {
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 10px;
    }
    .sys-block table { width: 100%; border-collapse: collapse; }
    .sys-block td { padding: 3px 0; font-size: 0.76rem; }
    .sys-block .td-key { color: var(--text-lo); }
    .sys-block .td-val { color: var(--text-mid); font-weight: 500; text-align: right; }
    .sys-block .td-val-green { color: var(--green); font-weight: 600; text-align: right; }

    /* ── Status indicator dot (no glow) ── */
    .status-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        margin-right: 5px;
        vertical-align: middle;
        flex-shrink: 0;
    }
    .status-dot-green  { background: var(--green); }
    .status-dot-red    { background: var(--red); }
    .status-dot-amber  { background: var(--amber); }
    .status-dot-blue   { background: var(--blue); }
    .status-dot-gray   { background: var(--text-lo); }

    /* ══ PAGE HEADER ══════════════════════════════════════════════════════════ */
    .page-header {
        margin-top: 24px;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }
    .page-header h1 {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: var(--text-hi) !important;
        margin: 0 0 4px !important;
        letter-spacing: -0.01em;
    }
    .page-header p {
        font-size: 0.83rem !important;
        color: var(--text-mid) !important;
        margin: 0 !important;
    }

    /* ══ SECTION LABEL ════════════════════════════════════════════════════════ */
    .section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }

    /* ══ KPI STRIP ════════════════════════════════════════════════════════════ */
    /* Streamlit metric cards are used for KPIs — neutral styling, no green border */

    /* ══ ALERT REPORT CARDS ═══════════════════════════════════════════════════ */
    .arc {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 12px 14px 12px 18px;
        margin-bottom: 8px;
        position: relative;
        transition: background-color 0.15s ease;
    }
    .arc:hover { background: var(--bg-surface2); }
    .arc::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        border-radius: var(--radius) 0 0 var(--radius);
    }
    .arc-critical::before { background: var(--red); }
    .arc-warning::before  { background: var(--amber); }
    .arc-safe::before     { background: var(--green); }

    /* Site name — prominent */
    .arc-name {
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-hi);
        line-height: 1.2;
        margin-bottom: 3px;
    }
    /* Location + waste type — subdued */
    .arc-meta {
        font-size: 0.74rem;
        color: var(--text-lo);
        margin-bottom: 6px;
    }
    /* Severity label — small, semantic color, no pill */
    .arc-severity {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    /* Time to critical — prominent operational number */
    .arc-ttc-wrap {
        display: flex;
        align-items: baseline;
        gap: 6px;
        margin: 6px 0;
    }
    .arc-ttc-num {
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1;
    }
    .arc-ttc-label {
        font-size: 0.72rem;
        color: var(--text-lo);
    }
    /* Gas / pollutant */
    .arc-gas {
        font-size: 0.75rem;
        color: var(--text-mid);
        margin-bottom: 6px;
    }
    /* Dispatch / action row */
    .arc-action-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid var(--border);
        font-size: 0.74rem;
        font-weight: 500;
    }

    /* ══ MAP HEADER ═══════════════════════════════════════════════════════════ */
    .map-header {
        margin-bottom: 8px;
    }
    .map-header .mh-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-hi);
        margin-bottom: 2px;
    }
    .map-header .mh-sub {
        font-size: 0.74rem;
        color: var(--text-lo);
    }
    .map-legend {
        display: flex;
        gap: 16px;
        margin-top: 6px;
        font-size: 0.72rem;
        color: var(--text-mid);
    }
    .map-legend span { display: flex; align-items: center; gap: 5px; }

    /* ══ OPENVINO RUNTIME CARD ════════════════════════════════════════════════ */
    .ov-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--blue);
        border-radius: var(--radius);
        padding: 12px 16px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .ov-body { flex: 1; }
    .ov-title {
        color: var(--text-lo);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .ov-device {
        color: var(--text-hi);
        font-size: 0.92rem;
        font-weight: 600;
    }
    .ov-latency {
        color: var(--text-mid);
        font-size: 0.78rem;
        margin-top: 2px;
    }

    /* ══ DECAY RESULT CARD ════════════════════════════════════════════════════ */
    .result-critical {
        background: var(--red-bg);
        border: 1px solid #3A1010;
        border-left: 3px solid var(--red);
        border-radius: var(--radius);
        padding: 18px 20px;
        margin: 12px 0;
    }
    .result-warning {
        background: var(--amber-bg);
        border: 1px solid #3A2200;
        border-left: 3px solid var(--amber);
        border-radius: var(--radius);
        padding: 18px 20px;
        margin: 12px 0;
    }
    .result-safe {
        background: var(--green-bg);
        border: 1px solid #0B2F18;
        border-left: 3px solid var(--green);
        border-radius: var(--radius);
        padding: 18px 20px;
        margin: 12px 0;
    }
    .result-level-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 3px;
    }
    .result-level-value {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 14px;
    }
    .kpi-row {
        display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px;
    }
    .kpi-cell {
        background: rgba(0,0,0,0.25);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 8px 14px;
        min-width: 110px;
    }
    .kpi-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 3px;
    }
    .kpi-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-hi);
    }

    /* ── Risk score bar ── */
    .risk-bar-wrap {
        background: rgba(0,0,0,0.3);
        border-radius: 3px;
        height: 6px;
        overflow: hidden;
    }
    .risk-bar-fill {
        height: 6px;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* ── Action recommendation box ── */
    .action-box {
        background: rgba(0,0,0,0.2);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 10px 14px;
        margin-top: 12px;
        font-size: 0.84rem;
        color: var(--text-mid);
        line-height: 1.6;
    }
    .action-box strong { color: var(--text-hi); }

    /* ══ UPLOAD IDLE PLACEHOLDER ══════════════════════════════════════════════ */
    .upload-idle {
        text-align: center;
        padding: 48px 24px;
        border: 1px dashed var(--border-mid);
        border-radius: var(--radius);
        background: var(--bg-surface);
        color: var(--text-lo);
    }
    .upload-idle .idle-icon { font-size: 2.2rem; margin-bottom: 10px; opacity: 0.6; }
    .upload-idle .idle-text { font-size: 0.88rem; color: var(--text-mid); }
    .upload-idle .idle-hint { font-size: 0.74rem; color: var(--text-lo); margin-top: 4px; }

    /* ══ RIGHT PANEL IDLE PLACEHOLDER ════════════════════════════════════════ */
    .right-idle {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 40px 20px;
        text-align: center;
        color: var(--text-lo);
    }
    .right-idle .ri-icon { font-size: 1.8rem; margin-bottom: 10px; opacity: 0.5; }
    .right-idle .ri-text { font-size: 0.85rem; color: var(--text-mid); }
    .right-idle .ri-hint { font-size: 0.74rem; color: var(--text-lo); margin-top: 6px; }

    /* ══ EXPORT NOTE ══════════════════════════════════════════════════════════ */
    .export-note {
        padding: 10px 14px;
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        font-size: 0.80rem;
        color: var(--text-mid);
        line-height: 1.5;
    }

    /* ══ LIVE MONITORING — FULL-VIEWPORT MAP CANVAS LAYOUT ═══════════════════ */

    /* ── 1. Zero-padding full-bleed canvas ─────────────────────────────────── */
    /* Strip every Streamlit wrapper so the map bleeds to all edges             */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }
    [data-testid="stMainBlockContainer"],
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        min-width: 100% !important;
    }

    /* ── 2. Folium / st_folium container — full viewport ───────────────────── */
    /* The iframe that st_folium injects must fill the whole viewport canvas.   */
    iframe {
        border: none !important;
        display: block !important;
        border-radius: 0 !important;
    }
    [data-testid="stIFrame"],
    .stIFrame,
    .element-container iframe {
        border-radius: 0 !important;
        overflow: hidden !important;
        box-shadow: none !important;
    }
    /* Guarantee the folium wrapper div stretches too */
    .folium-map {
        width: 100% !important;
        height: 88vh !important;
    }

    /* ── 3. TOP FILTER BAR ─────────────────────────────────────────────────── */
    .overlay-filter-bar {
        position: fixed;
        top: 15px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.90);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid #334155;
        border-radius: 50px;
        padding: 8px 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.55);
        white-space: nowrap;
    }
    .overlay-filter-bar .fb-brand {
        font-size: 0.78rem;
        font-weight: 700;
        color: #10B981;
        letter-spacing: 0.04em;
        margin-right: 8px;
        padding-right: 12px;
        border-right: 1px solid #334155;
    }
    .filter-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 14px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.15s ease;
        text-decoration: none;
    }
    .filter-pill-all {
        background: #10B981;
        color: #04120C;
        border-color: #10B981;
    }
    .filter-pill-critical {
        background: rgba(239,68,68,0.12);
        color: #EF4444;
        border-color: rgba(239,68,68,0.3);
    }
    .filter-pill-warning {
        background: rgba(245,158,11,0.12);
        color: #F59E0B;
        border-color: rgba(245,158,11,0.3);
    }
    .filter-pill-safe {
        background: rgba(16,185,129,0.12);
        color: #10B981;
        border-color: rgba(16,185,129,0.25);
    }
    .fb-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .fb-dot-critical { background: #EF4444; }
    .fb-dot-warning  { background: #F59E0B; }
    .fb-dot-safe     { background: #10B981; }
    .fb-dot-all      { background: #64748B; }

    /* ── 4. BOTTOM-LEFT ALERT CARD ──────────────────────────────────────────── */
    /* Strip Streamlit's wrapper div that bleeds a dark box behind the card      */
    [data-testid="stMarkdownContainer"]:has(.overlay-alert-card),
    div:has(> .overlay-alert-card) {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        border: none !important;
        overflow: visible !important;
    }

    /* ── Recent Alerts card (base declaration — see final override block) ───── */
    .dd-recent-alerts-card {
        position: fixed !important;
        bottom: 100px !important;
        left: 20px !important;
        z-index: 9999 !important;
        width: 310px !important;
        max-height: 340px !important;
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 14px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
        overflow-y: auto !important;
        display: block !important;
        visibility: visible !important;
    }

    /* Strip any Streamlit element-container chrome leaking behind the card */
    [data-testid="stElementContainer"]:has(.dd-recent-alerts-card),
    [data-testid="stMarkdownContainer"]:has(.dd-recent-alerts-card),
    div:has(> .dd-recent-alerts-card) {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
    }
    .overlay-alert-card {
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 999;
        width: 340px;
        background: rgba(15, 23, 42, 0.92);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 16px 16px 4px 16px; /* scroll child adds 20px bottom; 4px guards border-radius clip */
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        max-height: 420px;
        overflow: hidden;            /* clip the rounded corners */
        display: flex;
        flex-direction: column;
    }
    .overlay-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        flex-shrink: 0;
    }
    .overlay-card-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #64748B;
    }
    .overlay-card-badge {
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 50px;
        background: rgba(239,68,68,0.15);
        color: #EF4444;
        border: 1px solid rgba(239,68,68,0.3);
    }
    .overlay-alert-scroll {
        overflow-y: auto;
        overflow-x: hidden;          /* prevent any child from leaking sideways */
        flex: 1;
        padding-bottom: 20px;        /* breathing room above card bottom edge */
        scrollbar-width: thin;
        scrollbar-color: #334155 transparent;
    }
    .overlay-alert-scroll::-webkit-scrollbar { width: 3px; }
    .overlay-alert-scroll::-webkit-scrollbar-thumb {
        background: #334155; border-radius: 2px;
    }
    .oa-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid #1E293B;
    }
    .oa-item:last-child { border-bottom: none; }
    .oa-color-bar {
        width: 3px;
        border-radius: 3px;
        align-self: stretch;
        flex-shrink: 0;
        min-height: 48px;
    }
    .oa-body { flex: 1; min-width: 0; }
    .oa-name {
        font-size: 0.84rem;
        font-weight: 600;
        color: #E7EDF5;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 2px;
    }
    .oa-meta {
        font-size: 0.70rem;
        color: #64748B;
        margin-bottom: 5px;
    }
    .oa-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
    }
    .oa-ttc {
        font-size: 0.92rem;
        font-weight: 700;
        line-height: 1;
    }
    .oa-gas {
        font-size: 0.68rem;
        color: #8D9AAF;
    }
    .oa-status {
        font-size: 0.65rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 50px;
    }
    .oa-status-dispatched {
        background: rgba(16,185,129,0.12);
        color: #10B981;
        border: 1px solid rgba(16,185,129,0.3);
    }
    .oa-status-pending {
        background: rgba(74,85,104,0.15);
        color: #64748B;
        border: 1px solid #334155;
    }

    /* ── 5. RIGHT DETAIL DRAWER ─────────────────────────────────────────────── */
    .overlay-detail-drawer {
        position: fixed;
        top: 15px;
        right: 25px;
        z-index: 999;
        width: 380px;
        background: rgba(15, 23, 42, 0.90);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        max-height: calc(100vh - 30px);
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #334155 transparent;
    }
    .overlay-detail-drawer::-webkit-scrollbar { width: 3px; }
    .overlay-detail-drawer::-webkit-scrollbar-thumb {
        background: #334155; border-radius: 2px;
    }
    .dd-header {
        margin-bottom: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid #1E293B;
    }
    .dd-site-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #E7EDF5;
        margin-bottom: 3px;
    }
    .dd-district {
        font-size: 0.74rem;
        color: #64748B;
    }
    .dd-risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        margin-top: 8px;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .dd-badge-critical {
        background: rgba(239,68,68,0.15);
        color: #EF4444;
        border: 1px solid rgba(239,68,68,0.35);
    }
    .dd-badge-warning {
        background: rgba(245,158,11,0.15);
        color: #F59E0B;
        border: 1px solid rgba(245,158,11,0.35);
    }
    .dd-badge-safe {
        background: rgba(16,185,129,0.12);
        color: #10B981;
        border: 1px solid rgba(16,185,129,0.3);
    }
    .dd-kpi-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 12px;
    }
    .dd-kpi-cell {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 12px;
    }
    .dd-kpi-label {
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 4px;
    }
    .dd-kpi-value {
        font-size: 0.92rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .dd-risk-bar-wrap {
        background: #0F172A;
        border-radius: 3px;
        height: 5px;
        overflow: hidden;
        margin-top: 4px;
    }
    .dd-risk-bar-fill {
        height: 5px;
        border-radius: 3px;
    }
    .dd-notes-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.6;
        margin-top: 10px;
    }
    .dd-idle {
        text-align: center;
        padding: 40px 20px;
        color: #334155;
    }
    .dd-idle-icon { font-size: 2rem; margin-bottom: 10px; opacity: 0.5; }
    .dd-idle-text { font-size: 0.82rem; color: #475569; }
    .dd-idle-hint { font-size: 0.70rem; color: #334155; margin-top: 6px; }

    /* ── 6. BOTTOM METRIC BAR ───────────────────────────────────────────────── */
    .overlay-metric-bar {
        position: fixed;
        bottom: 15px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        display: flex;
        align-items: center;
        gap: 0;
        background: rgba(15, 23, 42, 0.90);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid #334155;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        overflow: hidden;
        white-space: nowrap;
    }
    .mb-cell {
        padding: 12px 22px;
        border-right: 1px solid #1E293B;
        text-align: center;
        min-width: 140px;
    }
    .mb-cell:last-child { border-right: none; }
    .mb-label {
        font-size: 0.60rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 3px;
    }
    .mb-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .mb-value-green { color: #10B981; }
    .mb-sub {
        font-size: 0.60rem;
        color: #94A3B8;
        margin-top: 2px;
    }
    .mb-pulse {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #10B981;
        margin-right: 4px;
        animation: pulse-green 2s infinite;
        vertical-align: middle;
    }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.8); }
    }

    /* ── 7. Streamlit internal elements — map page overrides ───────────────── */
    /* Hide the header bar's visual chrome so the map bleeds edge-to-edge,
       but do NOT use display:none — that kills the sidebar toggle button too.  */
    [data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
        box-shadow: none !important;
        z-index: 999999 !important;
        pointer-events: none !important;   /* pass clicks through to map */
    }

    /* ── Force sidebar toggle ALWAYS visible on top of the map canvas ──────── */
    /* Covers every testid variant across Streamlit versions + title selectors  */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    button[title="Open sidebar"],
    button[title="Close sidebar"] {
        z-index: 999999 !important;
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        background-color: #1E293B !important;
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
        color: #FFFFFF !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        pointer-events: all !important;    /* re-enable clicks on the button itself */
    }

    /* ══ ALERT CARD — DEFENSIVE OVERRIDES ════════════════════════════════════ */
    /* Streamlit injects `color: var(--text-mid) !important` on p / span / div  */
    /* inside stMarkdownContainer. These rules win via higher specificity so     */
    /* every text node inside the fixed alert card stays explicitly visible.     */

    /* Strip the dark Streamlit wrapper injected behind the fixed card markup */
    [data-testid="stMarkdownContainer"]:has(.dd-recent-alerts-card),
    [data-testid="stElementContainer"]:has(.dd-recent-alerts-card),
    div:has(> .dd-recent-alerts-card) {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    /* Force every text element inside the fixed alert card to honour
       the inline colour set on its parent — !important here beats the
       global Streamlit cascade that sets everything to --text-mid.   */
    .dd-recent-alerts-card div,
    .dd-recent-alerts-card span {
        color: inherit !important;
    }

    /* ── Legacy classes kept for non-map pages ─────────────────────────────── */
    .kpi-divider {
        height: 1px;
        background: var(--border);
        margin: 14px 0 18px;
    }
    .alerts-scroll-panel {
        max-height: 650px;
        overflow-y: auto;
        padding-right: 4px;
        scrollbar-width: thin;
        scrollbar-color: var(--border) transparent;
    }
    .alerts-scroll-panel::-webkit-scrollbar { width: 4px; }
    .alerts-scroll-panel::-webkit-scrollbar-thumb {
        background: var(--border); border-radius: 2px;
    }

    /* ══ LIVE MONITORING OVERLAY PANEL — POSITIONING CORRECTIONS ════════════ */

    /* ── A. Un-clip stMain so fixed children aren't trapped ─────────────────
       The full-bleed canvas rule earlier sets overflow:hidden on stMain which
       clips position:fixed descendants in some browser/Streamlit combos.
       Resetting to overflow:visible lets the fixed card paint freely.       */
    [data-testid="stMain"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        position: relative !important;
    }

    /* ── B. Recent Alerts card — position:fixed (viewport-relative) ─────────
       Using fixed (not absolute) guarantees the card is never clipped by any
       overflow:hidden ancestor.  left is offset past the expanded sidebar
       (Streamlit sidebar ≈ 336 px) so opening/closing the sidebar never
       covers the card; on narrow viewports it still starts at 20 px.       */
    .dd-recent-alerts-card {
        position: fixed !important;
        left: 20px !important;
        bottom: 100px !important;
        z-index: 9999 !important;
        width: 310px !important;
        max-height: 340px !important;
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 14px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
        overflow-y: auto !important;
        display: block !important;
        visibility: visible !important;
    }

    /* ── C. Wrapper transparency — strip Streamlit chrome around the card ─── */
    [data-testid="stMarkdownContainer"]:has(.dd-recent-alerts-card),
    [data-testid="stElementContainer"]:has(.dd-recent-alerts-card),
    div:has(> .dd-recent-alerts-card) {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    /* ── D. KPI bar — stays fixed at bottom-centre, z-index below card ──────
       bottom:15px + ~70px tall = top edge at ~85px → card's bottom:100px
       clears it with 15px breathing room.                                   */
    .overlay-metric-bar {
        z-index: 1000 !important;
    }

    /* ── E. Detail drawer — top-right, highest z but below alert card ───────
       Keep at z-index:1001 so it paints above the map but below the alerts. */
    .overlay-detail-drawer {
        z-index: 1001 !important;
    }

    /* ── F. Filter bar — top-centre, below everything interactive ─────────── */
    .overlay-filter-bar {
        z-index: 1002 !important;
    }

    /* ══ DECAY ASSESSMENT CARD — prevent clipping, full text render ══════════ */

    /* Un-clip Streamlit column and vertical block wrappers so the result card
       can grow to its natural height without being scissored.               */
    [data-testid="stColumn"],
    [data-testid="stVerticalBlock"] {
        overflow: visible !important;
        height: auto !important;
    }

    /* The three result-card variants: remove any height cap, ensure auto    */
    .result-critical,
    .result-warning,
    .result-safe {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        padding: 16px 20px !important;
        margin-bottom: 24px !important;
    }

    /* KPI cells inside result cards: allow natural height                   */
    .result-critical .kpi-cell,
    .result-warning  .kpi-cell,
    .result-safe     .kpi-cell {
        height: auto !important;
        overflow: visible !important;
    }

    /* Action recommendation box — force full text visibility                */
    .action-box {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #F1F5F9 !important;
        font-size: 0.84rem !important;
        line-height: 1.6 !important;
        margin-top: 12px !important;
    }
    .action-box strong {
        color: #FFFFFF !important;
    }

    /* kpi-value text inside result cards: high-contrast, fully visible      */
    .result-critical .kpi-value,
    .result-warning  .kpi-value,
    .result-safe     .kpi-value {
        color: var(--text-hi) !important;
        visibility: visible !important;
        opacity: 1 !important;
        display: block !important;
    }

    /* result-level-value (the CRITICAL / WARNING / SAFE headline)           */
    .result-level-value {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        margin-bottom: 14px !important;
    }

    /* Strip any Streamlit element-container that wraps a result card        */
    [data-testid="stElementContainer"]:has(.result-critical),
    [data-testid="stElementContainer"]:has(.result-warning),
    [data-testid="stElementContainer"]:has(.result-safe),
    [data-testid="stMarkdownContainer"]:has(.result-critical),
    [data-testid="stMarkdownContainer"]:has(.result-warning),
    [data-testid="stMarkdownContainer"]:has(.result-safe) {
        overflow: visible !important;
        height: auto !important;
        max-height: none !important;
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
        "css_class": "result-critical",
        "level_color": "#EF4444",
        "bar_color": "#EF4444",
        "action": (
            "<strong>Immediate action required.</strong> "
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
        "css_class": "result-warning",
        "level_color": "#F59E0B",
        "bar_color": "#F59E0B",
        "action": (
            "<strong>Schedule collection.</strong> "
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
        "css_class": "result-safe",
        "level_color": "#18B981",
        "bar_color": "#18B981",
        "action": (
            "<strong>Routine monitoring.</strong> "
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
    lo, hi = (91.0, 97.5) if "Seafood" in waste_type else \
              (86.0, 93.5) if "Vegetables" in waste_type else (88.0, 95.0)
    profile["confidence"] = round(random.uniform(lo, hi), 1)
    profile["latency_ms"] = latency_ms
    return profile


def _countdown_str(hours: int) -> str:
    target = datetime.now() + timedelta(hours=hours)
    return target.strftime("Critical by %H:%M  ·  %d %b %Y") + f"  ({hours}h window)"


def render_openvino_card(ov: dict) -> None:
    """Render the OpenVINO runtime status card."""
    device_label = ov["device"]
    latency      = ov["latency_ms"]
    ov_flag      = "Intel OpenVINO™ Optimized" if ov["available"] else "Simulation Mode"
    all_dev_str  = ", ".join(ov["all_devices"]) if ov["all_devices"] else "N/A"

    st.markdown(
        f"""
        <div class="ov-card">
            <div class="ov-body">
                <div class="ov-title">OpenVINO Runtime</div>
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
    css        = profile["css_class"]
    level      = profile["level"]
    lc         = profile["level_color"]
    hours      = profile["critical_hours"]
    score      = profile["risk_score"]
    bar_col    = profile["bar_color"]
    action     = profile["action"]
    gas        = profile["gas_risk"]
    conf       = profile["confidence"]
    lat        = profile["latency_ms"]
    countdown  = _countdown_str(hours)

    st.markdown(
        f"""
        <div class="{css}">
            <div class="result-level-label">Decay assessment</div>
            <div class="result-level-value" style="color:{lc};">{level}</div>
            <div class="kpi-row">
                <div class="kpi-cell">
                    <div class="kpi-label">Waste type</div>
                    <div class="kpi-value">{image_name}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">Critical window</div>
                    <div class="kpi-value" style="font-size:0.82rem;">{countdown}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">Gas risk</div>
                    <div class="kpi-value" style="font-size:0.84rem;">{gas}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">Confidence</div>
                    <div class="kpi-value">{conf}%</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">Inference</div>
                    <div class="kpi-value">{lat} ms</div>
                </div>
            </div>
            <div style="margin-bottom:10px;">
                <div class="kpi-label" style="margin-bottom:5px;">Odor risk score</div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div class="risk-bar-wrap" style="flex:1;">
                        <div class="risk-bar-fill"
                             style="width:{score}%;background:{bar_col};"></div>
                    </div>
                    <span style="color:{bar_col};font-weight:700;
                                 font-size:0.92rem;min-width:48px;">{score}/100</span>
                </div>
            </div>
            <div class="action-box">{action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── TPS Site Registry — global data ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

TPS_SITES = [
    {
        "name":           "TPS Pasar Minggu",
        "lat":            -6.2856,
        "lon":            106.8378,
        "waste_type":     "Seafood / Meat",
        "hours_to_decay": 8,
        "priority_score": 95,
        "risk_level":     "CRITICAL",
        "marker_color":   "red",
        "gas_risk":       "H₂S (High)",
        "action_status":  "Truck dispatched",
        "district":       "Jakarta Selatan",
        "notes":          "High seafood waste volume. H₂S odor detected by sensor array.",
    },
    {
        "name":           "TPS Pasar Senen",
        "lat":            -6.1754,
        "lon":            106.8454,
        "waste_type":     "Seafood / Meat",
        "hours_to_decay": 6,
        "priority_score": 98,
        "risk_level":     "CRITICAL",
        "marker_color":   "red",
        "gas_risk":       "H₂S (Critical)",
        "action_status":  "Truck dispatched",
        "district":       "Jakarta Pusat",
        "notes":          "Critical odor risk. Overflow detected. Immediate evacuation required.",
    },
    {
        "name":           "TPS Kebayoran Baru",
        "lat":            -6.2437,
        "lon":            106.7963,
        "waste_type":     "Vegetables / Fruits",
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
        "waste_type":     "Cooked Carbs / Dry Waste",
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
        "waste_type":     "Vegetables / Fruits",
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
        "waste_type":     "Cooked Carbs / Dry Waste",
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

_FOLIUM_ICON_MAP = {
    "red":    {"color": "red",    "icon": "exclamation-sign"},
    "orange": {"color": "orange", "icon": "warning-sign"},
    "green":  {"color": "green",  "icon": "ok-sign"},
}


# ══════════════════════════════════════════════════════════════════════════════
# ── Map & Table helpers ───────────────────────────────────────────────────────
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
        "CRITICAL": "background-color: #1C0A0A; color: #EF4444;",
        "WARNING":  "background-color: #1C1000; color: #F59E0B;",
        "SAFE":     "background-color: #061A10; color: #18B981;",
    }

    def row_style(row):
        return [STYLES.get(row["Risk Level"], "")] * len(row)

    return (
        df.style
        .apply(row_style, axis=1)
        .set_properties(**{
            "font-size":   "0.83rem",
            "font-weight": "500",
            "border":      "1px solid #26344A",
        })
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#111A2B"),
                ("color",            "#8D9AAF"),
                ("font-size",        "0.68rem"),
                ("font-weight",      "700"),
                ("letter-spacing",   "0.05em"),
                ("text-transform",   "uppercase"),
                ("border-bottom",    "1px solid #26344A"),
                ("padding",          "9px 12px"),
            ],
        }, {
            "selector": "td",
            "props": [("padding", "7px 12px")],
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
            "SAFE":     "#18B981",
        }.get(site["risk_level"], "#4A5568")

        popup_html = (
            f'<div style="font-family:\'Inter\',\'Segoe UI\',sans-serif;min-width:230px;'
            f'background:#111A2B;color:#E7EDF5;border-radius:6px;padding:14px 16px;'
            f'border-left:3px solid {risk_color};">'
            f'<b style="color:{risk_color};font-size:0.92rem;">{site["name"]}</b><br>'
            f'<span style="color:#4A5568;font-size:0.74rem;">{site["district"]}</span>'
            f'<hr style="border-color:#26344A;margin:8px 0;">'
            f'<table style="width:100%;font-size:0.78rem;border-collapse:collapse;">'
            f'<tr><td style="color:#4A5568;padding:3px 0;">Waste type</td>'
            f'<td style="color:#E7EDF5;text-align:right;">{site["waste_type"]}</td></tr>'
            f'<tr><td style="color:#4A5568;padding:3px 0;">Risk level</td>'
            f'<td style="color:{risk_color};font-weight:700;text-align:right;">{site["risk_level"]}</td></tr>'
            f'<tr><td style="color:#4A5568;padding:3px 0;">Hours to decay</td>'
            f'<td style="color:#E7EDF5;text-align:right;">{site["hours_to_decay"]}h</td></tr>'
            f'<tr><td style="color:#4A5568;padding:3px 0;">Gas risk</td>'
            f'<td style="color:#E7EDF5;text-align:right;">{site["gas_risk"]}</td></tr>'
            f'<tr><td style="color:#4A5568;padding:3px 0;">Priority score</td>'
            f'<td style="color:{risk_color};font-weight:700;text-align:right;">{site["priority_score"]}/100</td></tr>'
            f'<tr><td style="color:#4A5568;padding:3px 0;">Action</td>'
            f'<td style="color:#18B981;font-weight:600;text-align:right;">{site["action_status"]}</td></tr>'
            f'</table>'
            f'<div style="margin-top:10px;padding:6px 10px;background:#0B1220;'
            f'border-radius:4px;font-size:0.73rem;color:#8D9AAF;">{site["notes"]}</div>'
            f'</div>'
        )

        folium.Marker(
            location=[site["lat"], site["lon"]],
            popup=folium.Popup(
                folium.IFrame(popup_html, width=270, height=285),
                max_width=290,
            ),
            tooltip=f"{site['name']} — {site['risk_level']}",
            icon=folium.Icon(
                color=ic["color"],
                icon=ic["icon"],
                prefix="glyphicon",
            ),
        ).add_to(m)

    # ── Folium in-map legend removed ──────────────────────────────────────────
    # The bottom-left Risk Level legend (Critical / Warning / Normal) has been
    # removed because it duplicates the floating 'Recent Alerts' card rendered
    # by Streamlit, causing a stacked visual artefact at the same screen corner.
    return m


# ══════════════════════════════════════════════════════════════════════════════
# ── Sidebar — Navigation + System Status ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Brand ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='padding:4px 2px 12px;'>"
        "<div style='font-size:1.05rem;font-weight:700;color:#E7EDF5;"
        "letter-spacing:-0.01em;'>SmartRot AI</div>"
        "<div style='font-size:0.72rem;color:#4A5568;margin-top:2px;'>"
        "Jakarta Environmental Intelligence</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sb-rule'></div>", unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Control Room</div>',
                unsafe_allow_html=True)

    NAV_OPTIONS = [
        "Live Monitoring",
        "Decay Detection",
    ]
    active_page = st.radio(
        label="Navigation",
        options=NAV_OPTIONS,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<div class='sb-rule'></div>", unsafe_allow_html=True)

    # ── Deployment info ───────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Deployment</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.78rem;color:#8D9AAF;line-height:1.9;'>"
        "<span style='color:#4A5568;'>Environment</span><br>"
        "<span style='color:#E7EDF5;'>Pilot deployment</span><br>"
        "<span style='color:#4A5568;'>Region</span><br>"
        "<span style='color:#E7EDF5;'>DKI Jakarta</span><br>"
        "<span style='color:#4A5568;'>Sites monitored</span><br>"
        "<span style='color:#E7EDF5;'>6 TPS locations</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sb-rule'></div>", unsafe_allow_html=True)

    # ── Data sources ──────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Data sources</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.76rem;color:#4A5568;line-height:2.0;'>"
        "Edge camera feed (RTSP)<br>"
        "IoT sensor array<br>"
        "GeoJSON: DKI Jakarta TPS<br>"
        "DLH Jakarta open data"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sb-rule'></div>", unsafe_allow_html=True)

    # ── Intel hardware stats ──────────────────────────────────────────────────
    with st.expander("Intel hardware stats", expanded=False):
        st.markdown(
            """
            <table style="width:100%;font-size:0.76rem;border-collapse:collapse;">
                <tr>
                    <td style="color:#4A5568;padding:3px 0;">Framework</td>
                    <td style="color:#8D9AAF;font-weight:500;text-align:right;">Intel OpenVINO™ 2026</td>
                </tr>
                <tr>
                    <td style="color:#4A5568;padding:3px 0;">Mode</td>
                    <td style="color:#E7EDF5;font-weight:500;text-align:right;">Offline edge inference</td>
                </tr>
                <tr>
                    <td style="color:#4A5568;padding:3px 0;">Efficiency</td>
                    <td style="color:#18B981;font-weight:600;text-align:right;">3.2× lower wattage</td>
                </tr>
                <tr>
                    <td colspan="2" style="color:#4A5568;font-size:0.68rem;padding-bottom:3px;">
                        vs. PyTorch CPU baseline
                    </td>
                </tr>
                <tr>
                    <td style="color:#4A5568;padding:3px 0;">Devices</td>
                    <td style="color:#E7EDF5;font-weight:500;text-align:right;">NPU · GPU · CPU</td>
                </tr>
                <tr>
                    <td style="color:#4A5568;padding:3px 0;">Format</td>
                    <td style="color:#E7EDF5;font-weight:500;text-align:right;">OpenVINO IR (.xml/.bin)</td>
                </tr>
                <tr>
                    <td style="color:#4A5568;padding:3px 0;">Network</td>
                    <td style="color:#E7EDF5;font-weight:500;text-align:right;">Air-gapped</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='min-height:24px;'></div>", unsafe_allow_html=True)

    # ── System status block ───────────────────────────────────────────────────
    _ov_status = check_openvino_device()
    _device    = _ov_status["device"]
    _mode_txt  = "OpenVINO optimized" if _ov_status["available"] else "Simulation"
    st.markdown(
        f"""
        <div class="sys-block">
            <div class="sb-label">System status</div>
            <table>
                <tr>
                    <td class="td-key">Status</td>
                    <td class="td-val-green">
                        <span class="status-dot status-dot-green"></span>Operational
                    </td>
                </tr>
                <tr>
                    <td class="td-key">Runtime</td>
                    <td class="td-val">Intel OpenVINO IR</td>
                </tr>
                <tr>
                    <td class="td-key">Device</td>
                    <td class="td-val">{_device}</td>
                </tr>
                <tr>
                    <td class="td-key">Mode</td>
                    <td class="td-val">{_mode_txt}</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.66rem;color:#26344A;text-align:center;"
        "margin-top:10px;'>© 2026 SmartRot AI · Intel AI Challenge</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Main Canvas — page header ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_PAGE_TITLES = {
    NAV_OPTIONS[0]: (
        "DLH Jakarta Central Control Room",
        "Real-time TPS site monitoring across DKI Jakarta's 5 administrative cities. "
        "Click any map marker for a full site detail card.",
    ),
    NAV_OPTIONS[1]: (
        "Edge AI Decay Detector",
        "Upload a waste image, select its category, and run OpenVINO inference "
        "to obtain a decay timeline and odor risk assessment.",
    ),
}

# On the Live Monitoring page the floating filter bar replaces the page header.
# For all other pages the standard header is rendered normally.
if active_page != NAV_OPTIONS[0]:
    _title, _subtitle = _PAGE_TITLES[active_page]
    st.markdown(
        f"<div class='page-header'>"
        f"<h1>{_title}</h1>"
        f"<p>{_subtitle}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Live TPS Map & Control Room — Full-Viewport Canvas ─────────────────
# ══════════════════════════════════════════════════════════════════════════════
if active_page == NAV_OPTIONS[0]:

    # ── Pre-compute all values needed for overlay HTML ────────────────────────
    _URGENCY_COLOR = {
        "CRITICAL": "#EF4444",
        "WARNING":  "#F59E0B",
        "SAFE":     "#18B981",
    }
    _BAR_COLOR = {
        "CRITICAL": "#EF4444",
        "WARNING":  "#F59E0B",
        "SAFE":     "#10B981",
    }
    _BADGE_CLASS = {
        "CRITICAL": "dd-badge-critical",
        "WARNING":  "dd-badge-warning",
        "SAFE":     "dd-badge-safe",
    }

    n_critical = sum(1 for s in TPS_SITES if s["risk_level"] == "CRITICAL")
    n_warning  = sum(1 for s in TPS_SITES if s["risk_level"] == "WARNING")
    n_safe     = sum(1 for s in TPS_SITES if s["risk_level"] == "SAFE")
    n_total    = len(TPS_SITES)
    scan_time  = datetime.now().strftime("%H:%M")
    scan_date  = datetime.now().strftime("%d %b %Y")

    # ── 1. TOP FILTER BAR ─────────────────────────────────────────────────────
    _filter_bar_html = textwrap.dedent(f"""
        <div class="overlay-filter-bar">
            <span class="fb-brand">🗑️ SmartRot AI</span>
            <span class="filter-pill filter-pill-all">
                <span class="fb-dot fb-dot-all"></span>Semua TPS
            </span>
            <span class="filter-pill filter-pill-critical">
                <span class="fb-dot fb-dot-critical"></span>Kritis &nbsp;<b>{n_critical}</b>
            </span>
            <span class="filter-pill filter-pill-warning">
                <span class="fb-dot fb-dot-warning"></span>Warning &nbsp;<b>{n_warning}</b>
            </span>
            <span class="filter-pill filter-pill-safe">
                <span class="fb-dot fb-dot-safe"></span>Safe &nbsp;<b>{n_safe}</b>
            </span>
        </div>
    """).strip()
    st.markdown(_filter_bar_html, unsafe_allow_html=True)

    # ── 2. BOTTOM-LEFT ALERT CARD ─────────────────────────────────────────────
    _alert_items_html = ""
    for site in sorted(TPS_SITES, key=lambda x: x["priority_score"], reverse=True):
        risk    = site["risk_level"]
        bar_col = _BAR_COLOR[risk]
        # Decay window color: #EF4444 for CRITICAL, #F59E0B for WARNING, #10B981 for SAFE
        ttc_col = {"CRITICAL": "#EF4444", "WARNING": "#F59E0B", "SAFE": "#10B981"}[risk]

        # Action badge: green outline for dispatched, dark-gray for pending
        if site["action_status"] == "Truck dispatched":
            badge_style = (
                "display:inline-block; font-size:0.65rem; font-weight:600; "
                "padding:3px 9px; border-radius:50px; white-space:nowrap; "
                "background:rgba(16,185,129,0.12); color:#10B981 !important; "
                "border:1px solid rgba(16,185,129,0.4);"
            )
        else:
            badge_style = (
                "display:inline-block; font-size:0.65rem; font-weight:600; "
                "padding:3px 9px; border-radius:50px; white-space:nowrap; "
                "background:#1E293B; color:#94A3B8 !important; "
                "border:1px solid #334155;"
            )

        # Last item has no bottom border
        is_last     = site == sorted(TPS_SITES, key=lambda x: x["priority_score"], reverse=True)[-1]
        border_rule = "none" if is_last else "1px solid #263448"

        _alert_items_html += f"""
        <div style="display:flex; align-items:flex-start; gap:10px;
                    padding:10px 0; border-bottom:{border_rule};">
            <!-- left accent bar -->
            <div style="width:3px; border-radius:3px; align-self:stretch;
                        flex-shrink:0; min-height:52px;
                        background:{bar_col};"></div>
            <!-- body -->
            <div style="flex:1; min-width:0;">
                <!-- TPS name: #FFFFFF, font-weight:700 -->
                <div style="font-size:0.88rem; font-weight:700;
                             color:#FFFFFF !important;
                             white-space:nowrap; overflow:hidden;
                             text-overflow:ellipsis; margin-bottom:2px;">
                    {site['name']}
                </div>
                <!-- Subtext: location · waste type — #94A3B8, 0.75rem -->
                <div style="font-size:0.75rem; color:#94A3B8 !important;
                             margin-bottom:6px;">
                    {site['district']} &middot; {site['waste_type']}
                </div>
                <!-- Decay window / gas / action badge row -->
                <div style="display:flex; align-items:center;
                             justify-content:space-between; gap:6px;">
                    <!-- Decay window: #EF4444 critical, #F59E0B warning -->
                    <span style="font-size:0.92rem; font-weight:700;
                                  line-height:1; color:{ttc_col} !important;
                                  flex-shrink:0;">
                        {site['hours_to_decay']}h
                    </span>
                    <span style="font-size:0.68rem; color:#8D9AAF !important;
                                  flex:1; text-align:center; overflow:hidden;
                                  text-overflow:ellipsis; white-space:nowrap;">
                        {site['gas_risk']}
                    </span>
                    <span style="{badge_style}">
                        {site['action_status']}
                    </span>
                </div>
            </div>
        </div>"""

    _alert_card_html = textwrap.dedent(f"""
        <div class="dd-recent-alerts-card">
            <!-- Card header -->
            <div style="display:flex; align-items:center; justify-content:space-between;
                         margin-bottom:12px; flex-shrink:0;">
                <span style="font-size:0.72rem; font-weight:700; letter-spacing:0.07em;
                              text-transform:uppercase; color:#94A3B8 !important;">
                    RECENT ALERTS
                </span>
                <span style="font-size:0.65rem; font-weight:700; padding:3px 10px;
                              border-radius:50px; white-space:nowrap;
                              background:rgba(239,68,68,0.15); color:#EF4444 !important;
                              border:1px solid rgba(239,68,68,0.3);">
                    {n_critical} Kritikal / {n_critical} Critical
                </span>
            </div>
            <!-- Scrollable items -->
            <div style="overflow-y:auto; overflow-x:hidden; flex:1;
                         padding-bottom:8px;
                         scrollbar-width:thin;
                         scrollbar-color:#334155 transparent;">
                {_alert_items_html}
            </div>
        </div>
    """).strip()
    st.markdown(_alert_card_html, unsafe_allow_html=True)

    # ── 3. RIGHT DETAIL DRAWER ────────────────────────────────────────────────
    # Show the highest-priority site by default; users can click map markers
    # for inline Folium popups (full interactive detail).
    _top_site = sorted(TPS_SITES, key=lambda x: x["priority_score"], reverse=True)[0]
    _ts        = _top_site
    _rc        = _URGENCY_COLOR[_ts["risk_level"]]
    _bc        = _BAR_COLOR[_ts["risk_level"]]
    _bdg       = _BADGE_CLASS[_ts["risk_level"]]
    _score_pct = _ts["priority_score"]

    _detail_drawer_html = textwrap.dedent(f"""
<div class="overlay-detail-drawer">
<div class="dd-header">
<div class="dd-site-name">{_ts['name']}</div>
<div class="dd-district">
📍 {_ts['district']} &nbsp;·&nbsp; DKI Jakarta
</div>
<span class="dd-risk-badge {_bdg}">
<span style="width:6px;height:6px;border-radius:50%;
background:{_rc};display:inline-block;
flex-shrink:0;"></span>
{_ts['risk_level']}
</span>
</div>

<div class="dd-kpi-grid">
<div class="dd-kpi-cell" style="background:#1E293B;border:1px solid #334155;border-radius:10px;padding:10px 12px;">
<div class="dd-kpi-label" style="font-size:0.60rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#64748B;margin-bottom:4px;">Waste type</div>
<div class="dd-kpi-value" style="font-size:0.80rem;font-weight:700;color:#FFFFFF;">
{_ts['waste_type']}
</div>
</div>
<div class="dd-kpi-cell" style="background:#1E293B;border:1px solid #334155;border-radius:10px;padding:10px 12px;">
<div class="dd-kpi-label" style="font-size:0.60rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#64748B;margin-bottom:4px;">Hours to decay</div>
<div class="dd-kpi-value" style="font-size:0.92rem;font-weight:700;color:{_rc};">
{_ts['hours_to_decay']}h
</div>
</div>
<div class="dd-kpi-cell" style="background:#1E293B;border:1px solid #334155;border-radius:10px;padding:10px 12px;">
<div class="dd-kpi-label" style="font-size:0.60rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#64748B;margin-bottom:4px;">Gas risk</div>
<div class="dd-kpi-value" style="font-size:0.80rem;font-weight:700;color:#FFFFFF;">
{_ts['gas_risk']}
</div>
</div>
<div class="dd-kpi-cell" style="background:#1E293B;border:1px solid #334155;border-radius:10px;padding:10px 12px;">
<div class="dd-kpi-label" style="font-size:0.60rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#64748B;margin-bottom:4px;">Action</div>
<div class="dd-kpi-value" style="color:#10B981;font-size:0.80rem;font-weight:700;">
{_ts['action_status']}
</div>
</div>
</div>

<div style="font-size:0.60rem;font-weight:700;letter-spacing:0.06em;
text-transform:uppercase;color:#64748B;margin-bottom:6px;">
Priority score
</div>
<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
<div style="background:#0F172A;border-radius:3px;height:5px;
                            overflow:hidden;flex:1;">
                    <div style="height:5px;border-radius:3px;width:{_score_pct}%;
                                background:{_bc};"></div>
                </div>
                <span style="color:{_rc};font-weight:700;font-size:0.88rem;
                             min-width:40px;text-align:right;">
                    {_score_pct}/100
                </span>
            </div>

            <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;
                        padding:10px 12px;font-size:0.78rem;color:#94A3B8;line-height:1.6;
                        margin-top:10px;">
                {_ts['notes']}
            </div>

            <div style="margin-top:14px;padding:10px 12px;border-top:1px solid #1E293B;
background:#0F172A;border-radius:0 0 14px 14px;
font-size:0.60rem;color:#475569;text-align:center;letter-spacing:0.04em;">
TPS Site Detail &nbsp;·&nbsp; Click any map marker for inline popup
</div>
</div>
    """).strip()
    _detail_drawer_html = "\n".join([line.strip() for line in _detail_drawer_html.splitlines()])
    st.markdown(_detail_drawer_html, unsafe_allow_html=True)

    # ── 4. FULL-VIEWPORT MAP CANVAS ───────────────────────────────────────────
    # st_folium renders the Folium map inside an iframe. We set width=None so
    # Streamlit uses the full container width (now 100vw via CSS), and height
    # matches the 88vh target.  The page-header is suppressed via CSS so the
    # map truly fills edge-to-edge under all the floating cards.
    jakarta_map = build_folium_map()
    st_folium(
        jakarta_map,
        width=None,
        height=int(0.88 * 900),   # ~792px — matches 88vh on a 900px screen
        returned_objects=[],
    )

    # ── 5. BOTTOM METRIC BAR ──────────────────────────────────────────────────
    _metric_bar_html = textwrap.dedent(f"""
        <div class="overlay-metric-bar">
            <div class="mb-cell">
                <div class="mb-label">TPS Monitored</div>
                <div class="mb-value mb-value-green">{n_total}</div>
                <div class="mb-sub">DKI Jakarta pilot</div>
            </div>
            <div class="mb-cell">
                <div class="mb-label">Critical Alerts</div>
                <div class="mb-value" style="color:#EF4444;">{n_critical}</div>
                <div class="mb-sub">Immediate dispatch</div>
            </div>
            <div class="mb-cell">
                <div class="mb-label">Warning Sites</div>
                <div class="mb-value" style="color:#F59E0B;">{n_warning}</div>
                <div class="mb-sub">Scheduled pickup</div>
            </div>
            <div class="mb-cell">
                <div class="mb-label">Methane Avoided</div>
                <div class="mb-value mb-value-green">142 kg</div>
                <div class="mb-sub">vs. unmonitored baseline</div>
            </div>
            <div class="mb-cell">
                <div class="mb-label">Last Scan</div>
                <div class="mb-value" style="font-size:0.92rem;">{scan_time}</div>
                <div class="mb-sub">
                    <span class="mb-pulse"></span>{scan_date}
                </div>
            </div>
        </div>
    """).strip()
    st.markdown(_metric_bar_html, unsafe_allow_html=True)

    # ── 6. PRIORITY DISPATCH TABLE (collapsible, below the canvas) ───────────
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    with st.expander("Priority dispatch table — DLH fleet assignment",
                     expanded=False):
        st.caption(
            "Sorted by priority score (highest first).  "
            "Critical — immediate dispatch.  "
            "Warning — scheduled.  "
            "Safe — routine monitoring."
        )
        dispatch_df  = build_dispatch_dataframe()
        styled_table = _style_dispatch_table(dispatch_df)
        st.dataframe(styled_table, use_container_width=True, height=280)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        col_exp, col_note = st.columns([1, 3], gap="small")
        with col_exp:
            report_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            export_df  = dispatch_df.copy()
            export_df.insert(0, "Report Timestamp", report_ts)
            export_df.insert(1, "Pilot Region",     "DKI Jakarta")
            csv_buf    = io.StringIO()
            export_df.to_csv(csv_buf, index=False, encoding="utf-8")
            st.download_button(
                label="Export daily priority report (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"DLH_Priority_Report_"
                          f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_note:
            st.markdown(
                "<div class='export-note'>"
                "Includes TPS name, district, waste type, hours to decay, "
                "priority score, risk level, gas risk, and action status — "
                "timestamped for DLH Jakarta daily operational records."
                "</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: Edge AI Decay Detector ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif active_page == NAV_OPTIONS[1]:

    ov_info = check_openvino_device()

    # ── Session-state keys for this page ─────────────────────────────────────
    # "decay_image_bytes" — raw bytes of the uploaded file (persists across
    #                        reruns so the preview stays visible after the
    #                        button click rerun).
    # "decay_image_name"  — original filename for the caption.
    # "decay_profile"     — inference result dict; None until button clicked.
    # "decay_waste_type"  — the waste category used for the last run (kept so
    #                        the result card label stays consistent on reruns).
    if "decay_image_bytes" not in st.session_state:
        st.session_state.decay_image_bytes = None
    if "decay_image_name" not in st.session_state:
        st.session_state.decay_image_name  = None
    if "decay_profile" not in st.session_state:
        st.session_state.decay_profile     = None
    if "decay_waste_type" not in st.session_state:
        st.session_state.decay_waste_type  = None

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: image uploader + preview ───────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-label">Waste image input</div>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="Upload a TPS waste image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a photo of waste at the TPS site for decay analysis.",
            label_visibility="collapsed",
        )

        # Persist the upload in session_state so it survives the button-click
        # rerun.  Also clear any stale result when a new file is selected.
        if uploaded_file is not None:
            new_bytes = uploaded_file.getvalue()
            if new_bytes != st.session_state.decay_image_bytes:
                # New file — store it and wipe the previous result
                st.session_state.decay_image_bytes = new_bytes
                st.session_state.decay_image_name  = uploaded_file.name
                st.session_state.decay_profile     = None
                st.session_state.decay_waste_type  = None

        # Show preview if an image is stored in session_state
        if st.session_state.decay_image_bytes is not None:
            st.image(
                Image.open(io.BytesIO(st.session_state.decay_image_bytes)),
                caption=st.session_state.decay_image_name,
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

    # ── RIGHT: OV status + selector + run button + results ───────────────────
    with col_right:
        render_openvino_card(ov_info)

        st.markdown('<div class="section-label">Waste category</div>',
                    unsafe_allow_html=True)
        waste_type = st.selectbox(
            "Waste category",
            options=list(DECAY_PROFILES.keys()),
            index=0,
            help="Choose the category matching the image. "
                 "In production the OpenVINO model infers this automatically.",
            label_visibility="collapsed",
        )

        # Disable the button until an image is loaded
        image_ready = st.session_state.decay_image_bytes is not None
        run_analysis = st.button(
            "Run decay analysis",
            use_container_width=True,
            disabled=not image_ready,
            help="Upload a waste image first, then click to run OpenVINO inference.",
        )

        st.markdown("---")

        # ── Run inference ONLY on explicit button click ───────────────────────
        if run_analysis:
            pb = st.progress(0, text="Initialising OpenVINO runtime…")
            for pct, lbl in [
                (20,  "Loading IR model weights…"),
                (45,  "Pre-processing image tensor…"),
                (70,  f"Running inference on {ov_info['device']}…"),
                (90,  "Post-processing predictions…"),
                (100, "Analysis complete"),
            ]:
                time.sleep(0.25)
                pb.progress(pct, text=lbl)
            time.sleep(0.2)
            pb.empty()

            # Store result so it persists on subsequent reruns
            st.session_state.decay_profile    = simulate_inference(
                waste_type, ov_info["latency_ms"]
            )
            st.session_state.decay_waste_type = waste_type

        # ── Display stored result (if any) ────────────────────────────────────
        if st.session_state.decay_profile is not None:
            profile = st.session_state.decay_profile

            km1, km2, km3, km4 = st.columns(4)
            with km1:
                st.metric("Decay level",     profile["level"])
            with km2:
                st.metric("Critical window",
                          f"{profile['critical_hours']}h",
                          delta=f"-{profile['critical_hours']}h",
                          delta_color="inverse")
            with km3:
                st.metric("Odor risk score", f"{profile['risk_score']}/100")
            with km4:
                st.metric("Confidence",      f"{profile['confidence']}%")

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            render_result_card(profile, st.session_state.decay_waste_type)

        elif image_ready:
            # Image uploaded but analysis not yet run
            st.markdown(
                "<div class='right-idle'>"
                "<div class='ri-icon'>⚡</div>"
                "<div class='ri-text'>Image loaded. Press<br>"
                "<strong style='color:#E7EDF5;'>Run decay analysis</strong>"
                " to start OpenVINO inference.</div>"
                "<div class='ri-hint'>Inference runs on "
                f"{ov_info['device']}.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            # No image yet
            st.markdown(
                "<div class='right-idle'>"
                "<div class='ri-icon'>⚡</div>"
                "<div class='ri-text'>Select a waste category and press<br>"
                "<strong style='color:#E7EDF5;'>Run decay analysis</strong>"
                " to see the result here.</div>"
                "<div class='ri-hint'>Upload an image first to enable the button.</div>"
                "</div>",
                unsafe_allow_html=True,
            )