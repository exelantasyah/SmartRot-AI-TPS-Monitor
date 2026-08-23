# 🗑️ SmartRot AI
### Edge-AI TPS Decay & Odor Risk Monitor — DKI Jakarta

> **Intel AI Global Impact Festival 2026** · Pilot Project: DKI Jakarta

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-0071C5?logo=intel&logoColor=white)](https://docs.openvino.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

SmartRot AI is an **Edge-AI powered waste monitoring system** built for the Temporary Waste Collection Points (TPS) across DKI Jakarta's five administrative cities. It uses computer vision and Intel OpenVINO inference to detect waste decay levels and estimate odor risk in real time — giving the Dinas Lingkungan Hidup (DLH) Jakarta a data-driven tool for proactive waste fleet dispatch and public health risk mitigation.

The system runs inference directly on edge hardware (no cloud dependency), making it resilient, low-latency, and cost-efficient for city-scale deployment.

---

## ✨ Core Features

### 📷 Edge AI Decay Detector (TPS Level)
- **Real-time waste fill-level detection** from edge camera feeds or uploaded images
- **Decay classification** across three stages: `Fresh` → `Moderate` → `Critical`
- **Odor risk scoring** derived from visual texture and color degradation features
- Bounding-box overlays with per-detection confidence scores
- Automated alert generation logged per TPS site

### 🗺️ DLH Jakarta Central Control Room
- Interactive **Folium map** with GeoJSON overlay of all TPS locations across DKI Jakarta
- Color-coded risk markers: 🟢 Low · 🟡 Moderate · 🔴 Critical
- Clickable TPS popups with site detail cards (location, last scan, risk score)
- **Odor-risk heatmap** layer for density visualization across the city
- Fleet dispatch route suggestions for waste collection teams

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit (wide layout, dark enterprise theme) |
| **AI Inference** | Intel OpenVINO™ Runtime |
| **Computer Vision** | Custom IR model (`.xml` / `.bin`) |
| **Mapping** | Folium + streamlit-folium |
| **Data Handling** | Pandas, NumPy |
| **Language** | Python 3.10+ |
| **Target Hardware** | Intel CPU / iGPU / VPU (Edge Device) |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-org/SmartRot-AI-TPS-Monitor.git
cd SmartRot-AI-TPS-Monitor
```

### 2. Create and activate a virtual environment *(recommended)*
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
SmartRot-AI-TPS-Monitor/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── models/                 # OpenVINO IR model files (.xml / .bin)
├── data/                   # GeoJSON TPS location data
└── .streamlit/
    └── config.toml         # Streamlit theme config (optional)
```

---

## 🌏 Pilot Scope

| City | TPS Sites |
|---|---|
| Jakarta Pusat | ✅ Active |
| Jakarta Utara | ✅ Active |
| Jakarta Selatan | ✅ Active |
| Jakarta Barat | ✅ Active |
| Jakarta Timur | ✅ Active |

---

## 🤝 Team & Acknowledgements

Built for the **Intel AI Global Impact Festival 2026** by the SmartRot AI Team.  
Powered by **Intel OpenVINO™** for on-device AI inference.  
Geospatial data sourced from **DLH DKI Jakarta Open Data**.

---

*© 2026 SmartRot AI Team — Intel AI Innovation Challenge*
