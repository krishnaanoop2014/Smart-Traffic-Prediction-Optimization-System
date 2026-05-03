# 🚦 Smart Traffic Prediction & Optimization System

> Real-world smart city ML project — LSTM + Transformer for congestion forecasting, signal timing optimization, and route suggestion with an interactive Gradio dashboard.

[![Kaggle](https://img.shields.io/badge/Dataset-Metro%20I--94%20Traffic-blue?logo=kaggle)](https://www.kaggle.com/datasets/pooriamst/metro-interstate-traffic-volume)
[![Python](https://img.shields.io/badge/Python-3.10-yellow?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-orange?logo=gradio)](https://gradio.app)

---

## 📌 Problem Statement

Traffic congestion costs billions annually in lost productivity and fuel. This system uses deep learning to predict hourly traffic volume on I-94 (Minneapolis–Saint Paul), classify congestion levels, optimize traffic signal timing, and suggest optimal routes — all from a live interactive dashboard.

---

## 🏗️ Architecture

```
Raw Data
   │
   ▼
Feature Engineering ──► 24 features (cyclic time, lag, rolling stats, weather)
   │
   ▼
┌─────────────┐    ┌──────────────────────┐
│  BiLSTM     │    │  Transformer Encoder │
│ + Attention │    │  + Positional Enc.   │
└──────┬──────┘    └──────────┬───────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
          Volume Prediction
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
  Congestion  Signal     Route
  Level       Timing     Optimizer
  Classifier  Optimizer
                  │
                  ▼
          Gradio Dashboard + Folium Map
```

---

## 🔑 Key Features

| Feature | Details |
|---------|---------|
| **Models** | Bidirectional LSTM w/ Attention + Transformer Encoder |
| **Input** | 24 engineered features — lag, rolling stats, cyclic time, weather |
| **Output** | Traffic volume (vehicles/hr) + congestion level |
| **Signal Optimization** | Adaptive green/red timing based on predicted volume |
| **Route Suggestion** | 4 routes ranked by estimated travel time |
| **Visualization** | Folium heatmap with clickable segment popups |
| **Dashboard** | Gradio with real-time slider inputs + quick examples |

---

## 📊 Results

| Model | MAE | RMSE | R² | MAPE% |
|-------|-----|------|----|-------|
| BiLSTM + Attention | ~320 | ~480 | ~0.89 | ~8.5% |
| Transformer | ~340 | ~510 | ~0.87 | ~9.1% |

---

## 🗂️ Dataset

**Metro Interstate Traffic Volume** — hourly traffic on I-94 westbound (MN DOT)

- ~48,000 rows | 2012–2018
- Features: `traffic_volume`, `weather_main`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `holiday`

---

## 🚀 Run It

### On Kaggle (recommended)
1. Upload `smart_traffic_system.ipynb` to a new Kaggle notebook
2. Attach the [Metro I-94 Traffic dataset](https://www.kaggle.com/datasets/pooriamst/metro-interstate-traffic-volume)
3. Enable GPU accelerator
4. Run all cells — Gradio will output a public `share=True` link

### Locally
```bash
pip install -r requirements.txt
jupyter notebook smart_traffic_system.ipynb
```

---

## 📁 Project Structure

```
smart-traffic-system/
├── smart_traffic_system.ipynb   # Main notebook
├── requirements.txt
├── README.md
└── outputs/                     # Generated after run
    ├── eda.png
    ├── loss_curves.png
    ├── predictions.png
    ├── traffic_map.html
    ├── lstm_traffic.pt
    ├── transformer_traffic.pt
    ├── scalers.pkl
    └── model_results.csv
```

---

## 🧠 Technical Highlights

- **Cyclic encoding** of hour/day/month with sin/cos to avoid ordinality issues
- **Lag features** (1, 2, 3, 6, 12, 24 hrs) to capture temporal autocorrelation
- **Bidirectional LSTM** with self-attention pooling for context aggregation
- **Transformer** with learnable positional encoding
- **HuberLoss** training — robust to traffic spikes
- **Cosine annealing LR** with early stopping
- **Adaptive signal timing** — 5-level congestion → dynamic green time
- **Folium heatmap** with per-segment popups showing signal plan + best route

---

## 🛠️ Tech Stack

`PyTorch` · `Gradio` · `Folium` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

## 📬 Author

Built as a portfolio project demonstrating end-to-end ML engineering — from raw time series to deployed interactive dashboard.
