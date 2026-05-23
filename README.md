# Land Tenure Conflict Risk & Resolution Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ML-powered risk assessment platform for land parcels across Nigeria's Middle Belt and North-Central states, identifying high-conflict zones and supporting mediators, governors, and land registries in prioritising interventions.

---

## Problem Statement

Land conflicts kill thousands in Nigeria annually. The Middle Belt farmer-herder crisis and widespread land grabbing destabilise communities and deter investment. Without systematic risk scoring, mediators cannot prioritise scarce resources effectively.

---

## Features

| Feature | Description |
|---------|-------------|
| 4-Tier Risk Classifier | Gradient Boosting, Low / Medium / High / Critical |
| 5,000 Parcel Records | Covering 12 Middle Belt and North-Central states |
| Conflict Type Breakdown | Farmer-herder, boundary, land grabbing, ethnic claim |
| Documentation Score Analysis | Correlation between land registry quality and conflict risk |
| Parcel-Level Risk Scoring | Probability outputs for individual parcels |
| Interactive Dashboard | Streamlit app with spatial risk map |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | scikit-learn (Gradient Boosting), pandas |
| Geospatial | GeoPandas, Folium |
| Dashboard | Streamlit, Plotly |
| Data | NumPy, pandas |

---

## Project Structure

```
land-tenure-conflict/
├── src/
│   ├── data_generator.py    # Synthetic parcel dataset (5,000 records)
│   ├── model.py             # Gradient Boosting classifier, risk scoring
│   └── visualize.py         # Conflict map, risk distribution charts
├── streamlit_app.py         # Dashboard entry point
├── .streamlit/config.toml
├── requirements.txt
└── runtime.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/land-tenure-conflict.git
cd land-tenure-conflict
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Data Sources

- NLC (National Land Commission) parcel registry
- State land administration records (Plateau, Benue, Taraba, Kaduna, Niger)
- ACLED conflict event data
- GRID3 Nigeria administrative boundaries

---

## Author

**Momah Moses**, Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
