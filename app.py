"""
SmartRot AI: Edge-AI TPS Decay & Odor Risk Monitor (DKI Jakarta)
Pilot Project — DKI Jakarta
"""

import streamlit as st

# ── Page config must be the FIRST Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="SmartRot AI | DKI Jakarta TPS Monitor",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── AWS Console Dark-Enterprise Theme ─────────────────────────────────────────
AWS_DARK_CSS = """
<style>
    /* ── Global background ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f1923;
        color: #d1d5db;
        font-family: "Amazon Ember", "Helvetica Neue", Arial, sans-serif;
    }

    /* ── Top navigation bar ── */
    [data-testid="stHeader"] {
        background-color: #232F3E;
        border-bottom: 2px solid #FF9900;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1a2332;
        border-right: 1px solid #2d3f55;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #FF9900;
    }

    /* ── Main content area ── */
    [data-testid="stMainBlockContainer"] {
        background-color: #0f1923;
        padding-top: 1.5rem;
    }

    /* ── Tab bar ── */
    [data-testid="stTabs"] [role="tablist"] {
        background-color: #232F3E;
        border-radius: 6px 6px 0 0;
        padding: 4px 8px 0 8px;
        border-bottom: 2px solid #FF9900;
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: #9ba8b5;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 8px 18px;
        border-radius: 4px 4px 0 0;
        transition: color 0.2s, background-color 0.2s;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: #FF9900;
        background-color: #2d3f55;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #FF9900 !important;
        background-color: #0f1923 !important;
        border-bottom: 2px solid #FF9900;
    }

    /* ── Metric / info cards ── */
    [data-testid="stMetric"] {
        background-color: #1a2332;
        border: 1px solid #2d3f55;
        border-left: 3px solid #FF9900;
        border-radius: 6px;
        padding: 12px 16px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #FF9900;
        color: #0f1923;
        font-weight: 700;
        border: none;
        border-radius: 4px;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #e68a00;
        color: #0f1923;
    }

    /* ── Status badge helper classes ── */
    .badge-active {
        display: inline-block;
        background-color: #1e4620;
        color: #4ade80;
        border: 1px solid #4ade80;
        border-radius: 12px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-warn {
        display: inline-block;
        background-color: #3d2600;
        color: #FF9900;
        border: 1px solid #FF9900;
        border-radius: 12px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* ── Divider ── */
    hr {
        border-color: #2d3f55;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f1923; }
    ::-webkit-scrollbar-thumb { background: #2d3f55; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #FF9900; }
</style>
"""

st.markdown(AWS_DARK_CSS, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗑️ SmartRot AI")
    st.markdown("**Edge-AI TPS Decay &**  \n**Odor Risk Monitor**")
    st.markdown("---")

    # Pilot status badge
    st.markdown("### Project Status")
    st.markdown(
        '<span class="badge-active">● PILOT ACTIVE</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Pilot Project:** DKI Jakarta  \n"
        "**Scope:** Waste Collection Points (TPS)  \n"
        "**Region:** 5 Administrative Cities",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # App metadata
    st.markdown("### App Metadata")
    st.markdown(
        """
| Field | Value |
|---|---|
| Version | `1.0.0-alpha` |
| Model | OpenVINO IR |
| Framework | Streamlit |
| Inference | Intel Edge AI |
        """
    )
    st.markdown("---")

    # Quick links / info
    st.markdown("### Data Sources")
    st.markdown(
        "- 📡 Edge Camera Feed (RTSP)  \n"
        "- 🌡️ IoT Sensor Array  \n"
        "- 🗺️ GeoJSON: DKI Jakarta TPS  \n"
        "- 📊 DLH Jakarta Open Data"
    )
    st.markdown("---")

    st.markdown(
        "<small style='color:#9ba8b5;'>© 2026 SmartRot AI Team<br>"
        "Intel AI Innovation Challenge</small>",
        unsafe_allow_html=True,
    )

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#FF9900; margin-bottom:0;'>SmartRot AI</h1>"
    "<p style='color:#9ba8b5; font-size:1.05rem; margin-top:4px;'>"
    "Edge-AI TPS Decay &amp; Odor Risk Monitor &nbsp;·&nbsp; "
    "<span style='color:#4ade80;'>● DKI Jakarta Pilot</span></p>",
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
    st.markdown("### 📷 Edge AI Detector — TPS Level Analysis")
    st.info(
        "**[PLACEHOLDER]** This tab will host the real-time OpenVINO inference pipeline.  \n"
        "Capabilities planned:  \n"
        "- Live / uploaded image inference for waste fill-level detection  \n"
        "- TPS decay classification (Fresh → Moderate → Critical)  \n"
        "- Odor risk score estimation from visual features  \n"
        "- Confidence scores and bounding-box overlay  \n"
        "- Per-TPS alert generation and event logging"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="TPS Monitored", value="—", delta=None)
    with col2:
        st.metric(label="Critical Sites", value="—", delta=None)
    with col3:
        st.metric(label="Avg. Odor Risk", value="—", delta=None)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#9ba8b5; padding:60px 0;'>"
        "📷 Camera feed / image upload widget will render here."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Tab 2: DLH Jakarta Central Control Room ───────────────────────────────────
with tab_map:
    st.markdown("### 🗺️ DLH Jakarta Central Control Room")
    st.info(
        "**[PLACEHOLDER]** This tab will render an interactive Folium map.  \n"
        "Capabilities planned:  \n"
        "- GeoJSON overlay of all TPS locations across 5 DKI Jakarta cities  \n"
        "- Color-coded risk markers (green → yellow → red)  \n"
        "- Clickable popups with TPS detail cards  \n"
        "- Heatmap layer for odor-risk density  \n"
        "- Fleet dispatch route suggestions"
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Jakarta Pusat", "—")
    with col_b:
        st.metric("Jakarta Utara", "—")
    with col_c:
        st.metric("Jakarta Selatan", "—")
    with col_d:
        st.metric("Jakarta Barat / Timur", "—")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#9ba8b5; padding:60px 0;'>"
        "🗺️ Folium / streamlit-folium interactive map will render here."
        "</div>",
        unsafe_allow_html=True,
    )
