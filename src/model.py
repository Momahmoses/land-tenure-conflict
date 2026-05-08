"""
Land tenure conflict risk classification model.
Predicts conflict risk tier (Low/Medium/High/Critical) per parcel.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

FEATURES = [
    "area_ha", "parcel_value_m_naira", "documentation_score",
    "is_border_zone", "is_disputed", "multiple_claimants",
    "population_pressure", "drought_exposure", "migration_pressure",
    "prior_conflicts", "years_occupied", "govt_gazetting",
    "water_body_proximity_km", "land_use_enc", "tenure_enc",
]
TARGET = "conflict_risk_label"
MODEL_PATH = Path("assets/land_tenure_model.pkl")
META_PATH = Path("assets/land_tenure_meta.json")
RISK_ORDER = ["Low", "Medium", "High", "Critical"]


def _encode(df: pd.DataFrame):
    df = df.copy()
    le_lu = LabelEncoder()
    le_tenure = LabelEncoder()
    df["land_use_enc"] = le_lu.fit_transform(df["land_use"])
    df["tenure_enc"] = le_tenure.fit_transform(df["tenure_type"])
    return df, le_lu, le_tenure


def train(df: pd.DataFrame) -> tuple[GradientBoostingClassifier, dict]:
    df, le_lu, le_tenure = _encode(df)
    X = df[FEATURES].values
    y = df[TARGET].values

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    for train_idx, val_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        cv_accs.append(clf.score(X[val_idx], y[val_idx]))

    clf.fit(X, y)
    report = classification_report(y, clf.predict(X), output_dict=True)
    meta = {
        "cv_accuracy_mean": float(np.mean(cv_accs)),
        "cv_accuracy_std": float(np.std(cv_accs)),
        "train_accuracy": report["accuracy"],
        "feature_importances": dict(zip(FEATURES, clf.feature_importances_.tolist())),
        "land_use_classes": le_lu.classes_.tolist(),
        "tenure_classes": le_tenure.classes_.tolist(),
    }
    return clf, meta


def save_model(clf, meta: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))


def load_model():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())
