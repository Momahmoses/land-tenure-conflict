
# Land Tenure Conflict Risk & Resolution Platform
ML-powered risk assessment of land parcels across Nigeria's Middle Belt and North-Central states. Identifies high-conflict zones driven by farmer-herder clashes, land grabbing, boundary disputes, and population pressure. Supports mediators, state governors, and land registries in prioritising interventions.

---

## Features

- Gradient Boosting 4-tier risk classifier (Low / Medium / High / Critical)
- 5,000 parcel records across 12 states
- Conflict type breakdown: farmer-herder, boundary dispute, land grabbing, ethnic claim
- Documentation score analysis vs conflict risk correlation
- Parcel-level risk scorer with probability outputs

## Project Structure

```
land-tenure-conflict/
├── src/
│   ├── data_generator.py   # Synthetic parcel dataset (5,000 records)
│   └── model.py            # GBT risk classifier
├── streamlit_app.py        # 4-page Streamlit dashboard
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dataset

Synthetic dataset of 5,000 land parcels across Benue, Plateau, Taraba, Kaduna, Kogi, Niger, Nassarawa, Zamfara, Oyo, Ondo, Enugu, and Ebonyi states. Features include documentation score, tenure type, land use, population pressure, prior conflict count, and proximity to water bodies.

## Tech Stack

| Layer | Library |
|---|---|
| Dashboard | Streamlit |
| ML Model | scikit-learn GradientBoostingClassifier |
| Visualisation | Plotly Express + Mapbox |
| Data | NumPy / Pandas synthetic generator |

---

*Dataset is synthetic and generated for demonstration purposes.*
