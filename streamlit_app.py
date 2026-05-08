"""
Land Tenure Conflict Risk & Resolution Platform
================================================
ML-powered risk assessment of land parcels across Nigeria's Middle Belt and
North-Central states. Identifies high-conflict zones driven by farmer-herder
clashes, land grabbing, boundary disputes, and population pressure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import generate_land_tenure_dataset
from model import META_PATH, MODEL_PATH, load_model, save_model, train

RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}

st.set_page_config(
    page_title="Land Tenure Conflict | Nigeria",
    page_icon="🗺️",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return generate_land_tenure_dataset()


@st.cache_resource
def get_model():
    if MODEL_PATH.exists() and META_PATH.exists():
        return load_model()
    df = load_data()
    clf, meta = train(df)
    save_model(clf, meta)
    return clf, meta


df = load_data()
clf, meta = get_model()

st.sidebar.title("🗺️ Land Tenure Risk")
page = st.sidebar.radio("Navigate", ["Overview", "Risk Map", "Parcel Risk Scorer", "Model Performance"])
state_filter = st.sidebar.multiselect("State", sorted(df["state"].unique()), default=[])
land_use_filter = st.sidebar.multiselect("Land Use", sorted(df["land_use"].unique()), default=[])
tenure_filter = st.sidebar.multiselect("Tenure Type", sorted(df["tenure_type"].unique()), default=[])

filtered = df.copy()
if state_filter:
    filtered = filtered[filtered["state"].isin(state_filter)]
if land_use_filter:
    filtered = filtered[filtered["land_use"].isin(land_use_filter)]
if tenure_filter:
    filtered = filtered[filtered["tenure_type"].isin(tenure_filter)]

# ── overview ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Land Tenure Conflict Risk — Nigeria Middle Belt & North-Central")
    st.markdown(
        f"Analysing **{len(filtered):,} land parcels** across 12 states. "
        "Identifying high-conflict zones for mediation, gazetting, and legal intervention."
    )

    c1, c2, c3, c4 = st.columns(4)
    high_critical = filtered[filtered["conflict_risk_label"].isin(["High", "Critical"])]
    c1.metric("High/Critical Risk Parcels", f"{len(high_critical):,}", f"{len(high_critical)/len(filtered)*100:.1f}%")
    c2.metric("Disputed Parcels", f"{filtered['is_disputed'].sum():,}")
    c3.metric("Multiple Claimants", f"{filtered['multiple_claimants'].sum():,}")
    c4.metric("Avg Documentation Score", f"{filtered['documentation_score'].mean():.2f}")

    col_a, col_b = st.columns(2)
    with col_a:
        risk_counts = filtered["conflict_risk_label"].value_counts().reset_index()
        risk_counts.columns = ["risk", "count"]
        fig = px.bar(
            risk_counts, x="risk", y="count",
            color="risk", color_discrete_map=RISK_COLORS,
            title="Parcel Count by Conflict Risk Tier",
            category_orders={"risk": ["Low", "Medium", "High", "Critical"]},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        state_risk = filtered.groupby("state")["conflict_risk_score"].mean().reset_index()
        fig2 = px.bar(
            state_risk.sort_values("conflict_risk_score", ascending=False),
            x="state", y="conflict_risk_score",
            color="conflict_risk_score", color_continuous_scale="YlOrRd",
            title="Average Conflict Risk Score by State",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Conflict Type Distribution")
    ct = filtered[filtered["conflict_type"] != "None"]["conflict_type"].value_counts()
    fig3 = px.pie(ct.reset_index(), names="conflict_type", values="count",
                  title="Active Conflict Types", hole=0.4)
    st.plotly_chart(fig3, use_container_width=True)

# ── risk map ──────────────────────────────────────────────────────────────────
elif page == "Risk Map":
    st.title("Land Conflict Risk Map")
    color_var = st.selectbox(
        "Color by",
        ["conflict_risk_label", "conflict_risk_score", "documentation_score", "prior_conflicts"],
    )
    sample = filtered.sample(min(2000, len(filtered)), random_state=42)

    if color_var == "conflict_risk_label":
        fig = px.scatter_mapbox(
            sample, lat="latitude", lon="longitude",
            color="conflict_risk_label",
            color_discrete_map=RISK_COLORS,
            size="area_ha", size_max=18,
            hover_data=["state", "land_use", "tenure_type", "conflict_type", "prior_conflicts"],
            zoom=5, height=560,
            mapbox_style="carto-positron",
            title="Land Parcel Conflict Risk",
            category_orders={"conflict_risk_label": ["Low", "Medium", "High", "Critical"]},
        )
    else:
        scale = "RdYlGn" if color_var == "documentation_score" else "YlOrRd"
        fig = px.scatter_mapbox(
            sample, lat="latitude", lon="longitude",
            color=color_var,
            color_continuous_scale=scale,
            size="area_ha", size_max=18,
            hover_data=["state", "land_use", "tenure_type", "conflict_type"],
            zoom=5, height=560,
            mapbox_style="carto-positron",
            title=f"Land Parcels — {color_var}",
        )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.box(
            filtered, x="tenure_type", y="conflict_risk_score",
            color="tenure_type", title="Risk Score by Tenure Type",
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.scatter(
            filtered.sample(min(800, len(filtered)), random_state=3),
            x="documentation_score", y="conflict_risk_score",
            color="conflict_risk_label", color_discrete_map=RISK_COLORS,
            opacity=0.7, trendline="ols",
            title="Documentation Score vs Risk Score",
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── parcel risk scorer ────────────────────────────────────────────────────────
elif page == "Parcel Risk Scorer":
    st.title("Individual Parcel Risk Assessment")

    c1, c2, c3 = st.columns(3)
    with c1:
        land_use = st.selectbox("Land Use", meta["land_use_classes"])
        tenure_type = st.selectbox("Tenure Type", meta["tenure_classes"])
        area_ha = st.slider("Area (ha)", 0.1, 500.0, 5.0, step=0.5)
    with c2:
        value_m = st.slider("Parcel Value (₦M)", 0.5, 100.0, 10.0, step=0.5)
        doc_score = st.slider("Documentation Score", 0.0, 1.0, 0.4, step=0.05)
        is_border = st.checkbox("Border Zone", value=False)
        is_disputed = st.checkbox("Currently Disputed", value=False)
        multi_claim = st.checkbox("Multiple Claimants", value=False)
    with c3:
        pop_pressure = st.slider("Population Pressure", 0.0, 1.0, 0.5, step=0.05)
        drought = st.slider("Drought Exposure", 0.0, 1.0, 0.3, step=0.05)
        migration = st.slider("Migration Pressure", 0.0, 1.0, 0.35, step=0.05)
        prior_conflicts = st.slider("Prior Conflict Incidents", 0, 10, 1)
        years_occupied = st.slider("Years Occupied", 1, 50, 10)
        govt_gazetted = st.checkbox("Govt Gazetted", value=False)
        water_km = st.slider("Distance to Water Body (km)", 0.1, 50.0, 5.0)

    if st.button("Assess Parcel", type="primary"):
        lu_enc = meta["land_use_classes"].index(land_use) if land_use in meta["land_use_classes"] else 0
        tn_enc = meta["tenure_classes"].index(tenure_type) if tenure_type in meta["tenure_classes"] else 0
        row = np.array([[
            area_ha, value_m, doc_score,
            int(is_border), int(is_disputed), int(multi_claim),
            pop_pressure, drought, migration,
            prior_conflicts, years_occupied, int(govt_gazetted),
            water_km, lu_enc, tn_enc,
        ]])
        prediction = clf.predict(row)[0]
        proba = clf.predict_proba(row)[0]
        classes = clf.classes_

        colour = {"Low": "green", "Medium": "orange", "High": "red", "Critical": "red"}
        st.markdown(f"### :{colour.get(prediction, 'blue')}[{prediction} Risk]")

        prob_df = pd.DataFrame({"Risk Tier": classes, "Probability": proba})
        fig = px.bar(prob_df, x="Risk Tier", y="Probability",
                     color="Risk Tier", color_discrete_map=RISK_COLORS,
                     title="Risk Tier Probabilities",
                     category_orders={"Risk Tier": ["Low", "Medium", "High", "Critical"]})
        st.plotly_chart(fig, use_container_width=True)

# ── model performance ─────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Conflict Risk Classifier — Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("CV Accuracy", f"{meta['cv_accuracy_mean']:.4f}", f"±{meta['cv_accuracy_std']:.4f}")
    col2.metric("Training Accuracy", f"{meta['train_accuracy']:.4f}")

    fi = pd.Series(meta["feature_importances"]).sort_values(ascending=True)
    fig = px.bar(
        fi.reset_index(), x=0, y="index", orientation="h",
        title="Feature Importances (Gradient Boosting)",
        labels={0: "Importance", "index": "Feature"},
        color=0, color_continuous_scale="Oranges",
    )
    st.plotly_chart(fig, use_container_width=True)
