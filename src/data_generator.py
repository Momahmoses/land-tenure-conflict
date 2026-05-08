"""
Synthetic dataset generator for land tenure conflict risk analysis.
Each record is a land parcel with ownership, encroachment, and conflict history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_PARCELS = 5000

STATES = [
    ("Benue", 7.73, 8.52), ("Plateau", 9.22, 9.52), ("Taraba", 7.87, 11.47),
    ("Kaduna", 10.52, 7.44), ("Kogi", 7.80, 6.74), ("Niger", 9.93, 6.56),
    ("Nassarawa", 8.50, 8.54), ("Zamfara", 12.17, 6.66), ("Oyo", 7.85, 3.93),
    ("Ondo", 7.25, 5.20), ("Enugu", 6.46, 7.55), ("Ebonyi", 6.25, 8.01),
]

LAND_USE_TYPES = ["Farmland", "Grazing Reserve", "Forest Reserve", "Urban Fringe", "Wetland", "Mixed Use"]
TENURE_TYPES = ["Customary", "Statutory (C of O)", "Leasehold", "Informal Occupation", "Community Owned"]
CONFLICT_TYPES = ["None", "Farmer-Herder", "Boundary Dispute", "Land Grabbing", "Government Acquisition", "Ethnic Claim"]


def generate_land_tenure_dataset(n_parcels: int = N_PARCELS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for parcel_id in range(n_parcels):
        state, lat, lon = STATES[rng.integers(0, len(STATES))]
        land_use = LAND_USE_TYPES[rng.integers(0, len(LAND_USE_TYPES))]
        tenure_type = TENURE_TYPES[rng.integers(0, len(TENURE_TYPES))]
        conflict_type = CONFLICT_TYPES[rng.integers(0, len(CONFLICT_TYPES))]

        area_ha = max(0.1, float(rng.lognormal(1.5, 1.2)))
        parcel_value_m_naira = max(0.5, float(rng.lognormal(2.0, 1.0)))
        documentation_score = float(np.clip(
            (1.0 if tenure_type == "Statutory (C of O)" else rng.uniform(0, 0.7)), 0, 1
        ))

        # risk factors
        is_border_zone = int(rng.random() < 0.25)
        population_pressure = float(np.clip(rng.normal(0.5, 0.25), 0, 1))
        drought_exposure = float(np.clip(rng.normal(0.4, 0.2), 0, 1))
        migration_pressure = float(np.clip(rng.normal(0.35, 0.2) + (0.2 if land_use == "Grazing Reserve" else 0), 0, 1))
        prior_conflicts = int(rng.poisson(1.2 if conflict_type != "None" else 0.3))
        years_occupied = max(1, int(rng.normal(15, 10)))
        multiple_claimants = int(rng.random() < (0.35 if tenure_type in ("Customary", "Informal Occupation") else 0.08))
        is_disputed = int(conflict_type != "None")
        govt_gazetting = int(rng.random() < (0.5 if land_use in ("Grazing Reserve", "Forest Reserve") else 0.15))
        water_body_proximity_km = max(0.1, float(rng.exponential(5)))

        # conflict risk score
        risk_score = (
            0.25 * (1 - documentation_score) +
            0.20 * multiple_claimants +
            0.15 * population_pressure +
            0.15 * migration_pressure +
            0.10 * is_border_zone +
            0.10 * min(prior_conflicts / 5, 1) +
            0.05 * drought_exposure
        )
        conflict_risk_label = (
            "Low" if risk_score < 0.25 else
            "Medium" if risk_score < 0.50 else
            "High" if risk_score < 0.75 else "Critical"
        )

        records.append({
            "parcel_id": parcel_id, "state": state,
            "latitude": round(lat + rng.normal(0, 0.6), 6),
            "longitude": round(lon + rng.normal(0, 0.6), 6),
            "land_use": land_use, "tenure_type": tenure_type,
            "conflict_type": conflict_type, "area_ha": round(area_ha, 3),
            "parcel_value_m_naira": round(parcel_value_m_naira, 2),
            "documentation_score": round(documentation_score, 3),
            "is_border_zone": is_border_zone, "is_disputed": is_disputed,
            "multiple_claimants": multiple_claimants,
            "population_pressure": round(population_pressure, 3),
            "drought_exposure": round(drought_exposure, 3),
            "migration_pressure": round(migration_pressure, 3),
            "prior_conflicts": prior_conflicts,
            "years_occupied": years_occupied,
            "govt_gazetting": govt_gazetting,
            "water_body_proximity_km": round(water_body_proximity_km, 2),
            "conflict_risk_score": round(risk_score, 4),
            "conflict_risk_label": conflict_risk_label,
        })

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_land_tenure_dataset()
    path = output_dir / "land_tenure_data.csv"
    df.to_csv(path, index=False)
    return path
