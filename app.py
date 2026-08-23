"""
SmartRot AI: Edge-AI TPS Decay & Odor Risk Monitor (DKI Jakarta)
Pilot Project — DKI Jakarta
"""

import random
import time
from datetime import datetime, timedelta

import numpy as np
import streamlit as st
from PIL import Image

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

    /* ── OpenVINO status card ── */
    .ov-card {
        background-color: #0d1f33;
        border: 1px solid #0071C5;
        border-left: 4px solid #0071C5;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 1rem;
    }
    .ov-card .ov-title {
        color: #60a5fa;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .ov-card .ov-device {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .ov-card .ov-latency {
        color: #4ade80;
        font-size: 0.85rem;
        margin-top: 2px;
    }

    /* ── Result alert cards ── */
    .alert-critical {
        background-color: #2d0a0a;
        border: 1px solid #ef4444;
        border-left: 5px solid #ef4444;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #2d1a00;
        border: 1px solid #FF9900;
        border-left: 5px solid #FF9900;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 1rem 0;
    }
    .alert-safe {
        background-color: #0a2010;
        border: 1px solid #4ade80;
        border-left: 5px solid #4ade80;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 1rem 0;
    }
    .alert-critical h3 { color: #ef4444; margin: 0 0 8px 0; font-size: 1.1rem; }
    .alert-warning  h3 { color: #FF9900; margin: 0 0 8px 0; font-size: 1.1rem; }
    .alert-safe     h3 { color: #4ade80; margin: 0 0 8px 0; font-size: 1.1rem; }
    .alert-row { display: flex; gap: 32px; flex-wrap: wrap; margin-top: 10px; }
    .alert-item label {
        display: block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #9ba8b5;
        margin-bottom: 2px;
    }
    .alert-item span {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .action-box {
        background-color: #1a2332;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 12px;
        font-size: 0.9rem;
        color: #d1d5db;
        border: 1px solid #2d3f55;
    }
    .action-box strong { color: #FF9900; }

    /* ── Risk score bar ── */
    .risk-bar-wrap {
        margin-top: 8px;
        background: #1a2332;
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
    }
    .risk-bar-fill {
        height: 10px;
        border-radius: 4px;
        transition: width 0.6s ease;
    }
</style>
"""

st.markdown(AWS_DARK_CSS, unsafe_allow_html=True)


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
        "badge_color": "#FF9900",
        "bar_color": "#FF9900",
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
        "badge_color": "#4ade80",
        "bar_color": "#4ade80",
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
    """Render the OpenVINO status card."""
    device_label = ov["device"]
    latency      = ov["latency_ms"]
    ov_flag      = "Intel OpenVINO™ Optimized" if ov["available"] else "Simulation Mode"
    all_dev_str  = ", ".join(ov["all_devices"]) if ov["all_devices"] else "N/A"

    st.markdown(
        f"""
        <div class="ov-card">
            <div class="ov-title">⚡ OpenVINO Runtime Status</div>
            <div class="ov-device">Active Device: {device_label}</div>
            <div class="ov-latency">
                {latency} ms · {ov_flag}
                &nbsp;|&nbsp; All detected: {all_dev_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(profile: dict, image_name: str) -> None:
    """Render the full decay analysis result card."""
    css     = profile["css_class"]
    level   = profile["level"]
    hours   = profile["critical_hours"]
    score   = profile["risk_score"]
    bar_col = profile["bar_color"]
    action  = profile["action"]
    gas     = profile["gas_risk"]
    gas_ico = profile["gas_icon"]
    conf    = profile["confidence"]
    lat     = profile["latency_ms"]
    countdown = _countdown_str(hours)

    st.markdown(
        f"""
        <div class="{css}">
            <h3>{gas_ico} Decay Level: {level}</h3>
            <div class="alert-row">
                <div class="alert-item">
                    <label>Waste Type</label>
                    <span>{image_name}</span>
                </div>
                <div class="alert-item">
                    <label>Critical Window</label>
                    <span>{countdown}</span>
                </div>
                <div class="alert-item">
                    <label>Gas Risk</label>
                    <span>{gas}</span>
                </div>
                <div class="alert-item">
                    <label>Confidence</label>
                    <span>{conf}%</span>
                </div>
                <div class="alert-item">
                    <label>Inference Time</label>
                    <span>{lat} ms</span>
                </div>
            </div>
            <div style="margin-top:10px;">
                <label style="font-size:0.72rem;font-weight:700;letter-spacing:0.07em;
                              text-transform:uppercase;color:#9ba8b5;">
                    Odor Risk Score
                </label>
                <div style="display:flex;align-items:center;gap:10px;margin-top:4px;">
                    <div class="risk-bar-wrap" style="flex:1;">
                        <div class="risk-bar-fill"
                             style="width:{score}%;background:{bar_col};"></div>
                    </div>
                    <span style="color:{bar_col};font-weight:700;min-width:36px;">{score}/100</span>
                </div>
            </div>
            <div class="action-box">{action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown("### 📷 Edge AI Detector — TPS Waste Decay Analysis")

    # ── OpenVINO device probe (cached) ────────────────────────────────────────
    ov_info = check_openvino_device()
    render_openvino_card(ov_info)

    st.markdown("---")

    # ── Input layout: uploader left, selector right ───────────────────────────
    col_upload, col_select = st.columns([2, 1], gap="large")

    with col_upload:
        st.markdown("#### 📂 Upload Waste Image")
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
                caption=f"📷 Uploaded: {uploaded_file.name}",
                use_column_width=True,
            )

    with col_select:
        st.markdown("#### 🗂️ Waste Category")
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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<small style='color:#9ba8b5;'>"
            "The classifier uses the selected category to determine "
            "decay timeline and gas risk profile. In production, "
            "this is inferred automatically by the OpenVINO model."
            "</small>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
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
            "<div style='text-align:center;color:#9ba8b5;padding:48px 0;"
            "border:1px dashed #2d3f55;border-radius:8px;'>"
            "<div style='font-size:2.5rem;'>📷</div>"
            "<div style='margin-top:8px;font-size:1rem;'>Upload a waste image and press"
            " <strong style='color:#FF9900;'>Run Decay Analysis</strong> to begin.</div>"
            "<div style='font-size:0.82rem;margin-top:6px;color:#4a5568;'>"
            "Supported formats: JPG · PNG · WEBP</div>"
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
