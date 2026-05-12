# ATFOS — AI-Driven Traffic Flow Optimization System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-orange?style=flat-square)](https://github.com/Chaitanya1914/Traffic-Flow-Optimization-System)

**Intelligent traffic signal optimization using machine learning predictions and real-time vehicle detection**

[Quick Start](#-quick-start) • [Features](#-features) • [Usage](#-usage) • [Architecture](#-architecture)

</div>

---

## 🎯 Overview

ATFOS is a hybrid AI system that combines **historical traffic intelligence** with **real-time computer vision** to dynamically optimize traffic signal timings. It predicts traffic flow using machine learning and detects vehicles in real-time to make adaptive signal decisions.

**Key Innovation**: Fusion of Random Forest predictions (historical patterns) + YOLOv8 detection (live vehicle count) → Smart signal timing recommendations.

---

## ⚡ Features

| Feature | Technology |
|---------|-----------|
| **Traffic Speed Prediction** | Random Forest Regressor |
| **Real-Time Vehicle Detection** | YOLOv8 (Nano) |
| **Signal Decision Logic** | ML + CV Fusion Engine |
| **Interactive Dashboard** | Streamlit Web UI |
| **Feature Importance Analysis** | Matplotlib Visualization |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Webcam or video file

### Installation

```bash
# Clone repository
git clone https://github.com/Chaitanya1914/Traffic-Flow-Optimization-System.git
cd Traffic-Flow-Optimization-System

# Install dependencies
pip install -r requirements.txt

# Train model (one-time)
python train_model.py

# Option A: Run main system with video
python atfos_master.py

# Option B: Launch web dashboard
streamlit run atfos_web.py
```

---

## 📋 How It Works

### System Pipeline

```
Video Input (CCTV/Webcam)
    ↓
[YOLOv8 Vehicle Detection] → Live vehicle count
    ↓
[Random Forest Model] → Predict historical speed
    ↓
[Decision Logic] → Signal timing recommendation
    ↓
Output: "HEAVY TRAFFIC: 60s Green" (with confidence)
```

### Decision Rules

| Condition | Signal Timing | Status |
|-----------|--------------|--------|
| Count > 15 OR Speed < 20 km/h | 60s Green | 🔴 Heavy |
| Count > 5 | 45s Green | 🟡 Moderate |
| Count ≤ 5 | 20s Green | 🟢 Low |

---

## 📊 Technical Stack

**Core Libraries**:
- `scikit-learn` — Machine learning (Random Forest)
- `ultralytics` — YOLOv8 vehicle detection
- `streamlit` — Web dashboard
- `opencv-python` — Video processing
- `pandas` — Data handling
- `joblib` — Model persistence

---

## 🎮 Usage Modes

### Mode 1: Real-Time Analysis (CLI)
```bash
python atfos_master.py
# Processes video, overlays detections and signals
# Press 'q' to exit
```

### Mode 2: Interactive Dashboard
```bash
streamlit run atfos_web.py
# Opens browser dashboard with live metrics
# Click "Start System" button to begin
```

### Mode 3: Train Custom Model
```bash
python train_model.py
# Trains on delhi_traffic_features.csv
# Outputs: atfos_model.pkl, feature_importance.png
```

---

## 📂 Project Structure

```
ATFOS/
├── atfos_master.py              # Core inference engine
├── atfos_web.py                 # Streamlit dashboard
├── train_model.py               # Model training pipeline
├── atfos_vision.py              # Vision utilities
├── delhi_traffic_features.csv   # Training dataset
├── atfos_model.pkl              # Pre-trained model
├── yolov8n.pt                   # YOLOv8 weights
└── videoplayback.mp4            # Sample video
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Model Type** | Random Forest Regressor |
| **Features** | 7 (time, weather, location, etc.) |
| **Training Samples** | 2,000+ records |
| **Inference Speed** | <50ms per frame |
| **Vehicle Detection** | Real-time (30 FPS) |

---

## 💡 Use Cases

- 🏙️ **Smart City Traffic Management** — Adaptive signal control
- 🚗 **Urban Congestion Reduction** — Data-driven timing optimization
- 🚨 **Emergency Response** — Priority lane clearance
- 📊 **Traffic Analytics** — Pattern recognition and forecasting
- 🎓 **Research & Academia** — ML + CV fusion studies

---

## 🔧 Configuration

Edit signal timing in `atfos_master.py`:

```python
def make_decision(live_count, hist_speed):
    if live_count > 15 or hist_speed < 20:
        return "HEAVY TRAFFIC: Green Light for 60s", (0, 0, 255)
    elif live_count > 5:
        return "MODERATE TRAFFIC: Green Light for 45s", (0, 255, 255)
    else:
        return "LOW TRAFFIC: Green Light for 20s", (0, 255, 0)
```

---

## 📈 Future Enhancements

- [ ] Real-time city-wide CCTV feed integration
- [ ] Deep learning traffic prediction (LSTM)
- [ ] Emergency vehicle prioritization
- [ ] Cloud deployment (AWS/GCP)
- [ ] Mobile app for traffic visualization
- [ ] IoT signal hardware integration

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details

---

## 👤 Author

**Chaitanya Singh**  
B.Tech CSE (AI & ML)  
[GitHub](https://github.com/Chaitanya1914) | [Email](mailto:chaitanya1914dev@gmail.com)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

---

<div align="center">

**Made with ❤️ for smarter cities**

[⭐ Star](https://github.com/Chaitanya1914/Traffic-Flow-Optimization-System) • [🐛 Issues](https://github.com/Chaitanya1914/Traffic-Flow-Optimization-System/issues) • [💬 Discussions](https://github.com/Chaitanya1914/Traffic-Flow-Optimization-System/discussions)

</div>
