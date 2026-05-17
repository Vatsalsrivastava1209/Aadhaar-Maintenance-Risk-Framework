"""Streamlit demo: Aadhaar Identity Maintenance Risk Framework.

Run:
    streamlit run app.py

This is a slim, runnable demo built on the same `utils` modules used by the
notebook — so anything you change in `utils/` is reflected here automatically.
The app loads the raw enrolment/biometric/demographic CSV folders, builds the
district-level risk index, and lets the user explore archetypes, top-risk
districts, and the choropleth.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.config import CONFIG
from utils.helpers import (
    calculate_balance_score,
    calculate_risk_index,
    naive_baseline_risk,
    risk_per_capita,
)

# Demo-mode fallback: when the full datasets/ folders are not present (typical
# fresh-clone case, since they're gitignored for size), we boot off a small
# committed sample under datasets/sample/. The app shows a banner so a recruiter
# clicking through a resume link sees something instead of a blank "no data" screen.
SAMPLE_BASE = Path("datasets/sample")


# ---------------------------------------------------------------------------
# Data loading (cached so reruns are instant)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_csv_folder(folder: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        # Empty frame from a missing folder — .str accessor would fail on the
        # default RangeIndex.
        return df
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


@st.cache_data(show_spinner="Building district-level risk frame...")
def build_district_frame() -> tuple[pd.DataFrame, dict, bool]:
    base = Path("datasets")
    enrol = _clean_columns(_load_csv_folder(str(base / "api_data_aadhar_enrolment")))
    bio = _clean_columns(_load_csv_folder(str(base / "api_data_aadhar_biometric")))
    demo = _clean_columns(_load_csv_folder(str(base / "api_data_aadhar_demographic")))

    demo_mode = False
    if enrol.empty or demo.empty:
        # Fall back to the committed sample so the app boots out-of-the-box.
        enrol = _clean_columns(_load_csv_folder(str(SAMPLE_BASE / "api_data_aadhar_enrolment")))
        bio = _clean_columns(_load_csv_folder(str(SAMPLE_BASE / "api_data_aadhar_biometric")))
        demo = _clean_columns(_load_csv_folder(str(SAMPLE_BASE / "api_data_aadhar_demographic")))
        demo_mode = True
        if enrol.empty or demo.empty:
            st.error(
                "Neither `datasets/` nor `datasets/sample/` contains the expected folders. "
                "See the notebook for the expected layout."
            )
            st.stop()

    for df in (enrol, bio, demo):
        df["pincode"] = df["pincode"].astype(str).str.strip()
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.strip().str.upper()

    pincode_csv = (
        str(SAMPLE_BASE / "pincode_directory_sample.csv")
        if demo_mode and (SAMPLE_BASE / "pincode_directory_sample.csv").exists()
        else CONFIG["files"]["pincode_mapping"]
    )
    pincode_map = (
        pd.read_csv(pincode_csv, usecols=["pincode", "district", "state"], dtype={"pincode": str})
        .drop_duplicates(subset="pincode")
        .set_index("pincode")
    )

    def attach(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        out_cols = [c for c in ("district", "state") if c in pincode_map.columns]
        df = df.drop(columns=[c for c in out_cols if c in df.columns])
        return df.merge(pincode_map[out_cols], left_on="pincode", right_index=True, how="left")

    enrol = attach(enrol)
    demo = attach(demo)

    enrol["total_enrolments"] = enrol[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
    demo["total_updates"] = demo[["demo_age_5_17", "demo_age_17_"]].sum(axis=1)

    district_df = (
        enrol.groupby("district")["total_enrolments"]
        .sum()
        .to_frame()
        .join(demo.groupby("district")["total_updates"].sum().to_frame(), how="inner")
        .fillna(0)
    )
    district_df["update_rate"] = (
        (district_df["total_updates"] / district_df["total_enrolments"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    balance = calculate_balance_score(demo.assign(group=demo["district"]))
    district_df = district_df.join(balance).reset_index()
    district_df = calculate_risk_index(district_df)
    district_df = district_df[
        district_df["total_enrolments"] > CONFIG["thresholds"]["min_enrol_for_analysis"]
    ].copy()
    district_df["baseline_risk"] = naive_baseline_risk(district_df)

    district_df["risk_per_capita"] = risk_per_capita(district_df).values

    with open(CONFIG["files"]["district_geojson"], encoding="utf-8") as f:
        geo = json.load(f)
    return district_df, geo, demo_mode


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="UIDAI Aadhaar Risk Framework", layout="wide")
st.title("Aadhaar Identity Maintenance Risk Framework")
st.caption(
    "District-level prioritization engine using Aadhaar enrolment / update data. "
    "Built from the same `utils.helpers` module as the analysis notebook."
)

district_df, districts_geo, demo_mode = build_district_frame()

if demo_mode:
    st.warning(
        "**Demo mode** — running on a small committed sample under `datasets/sample/` "
        "because the full UIDAI extracts under `datasets/` were not found. "
        "The risk index, archetype logic, and map are real; the numbers are not "
        "production-grade. Drop the real extracts into `datasets/` to switch.",
        icon=":material/info:",
    )

with st.sidebar:
    st.header("Filters")
    risk_tiers = st.multiselect(
        "Risk tier",
        options=district_df["risk_level"].dropna().unique().tolist(),
        default=district_df["risk_level"].dropna().unique().tolist(),
    )
    metric_choice = st.selectbox(
        "Map metric",
        options=["identity_maintenance_risk", "risk_per_capita", "p_failure", "impact", "update_rate"],
        index=0,
        help=(
            "`identity_maintenance_risk` is the composite (P(failure) x Impact). "
            "`risk_per_capita` strips the Impact (volume) term so large districts "
            "stop dominating the map — useful as a secondary triage view."
        ),
    )
    st.markdown("---")
    st.markdown(
        "**Composite weights** (P(failure) × Impact framing):\n"
        f"- update_rate: `{CONFIG['risk_weights']['p_failure']['update_rate']}`\n"
        f"- balance:     `{CONFIG['risk_weights']['p_failure']['balance']}`\n"
        f"- p_failure:   `{CONFIG['risk_weights']['composite']['p_failure']}`\n"
        f"- impact:      `{CONFIG['risk_weights']['composite']['impact']}`"
    )

view = district_df[district_df["risk_level"].isin(risk_tiers)].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Districts analysed", f"{len(view):,}")
col2.metric("Median risk score", f"{view['identity_maintenance_risk'].median():.3f}")
col3.metric("High-risk districts", f"{(view['risk_level'] == 'High Risk').sum():,}")
col4.metric("Total enrolments (M)", f"{view['total_enrolments'].sum() / 1e6:.1f}")

# Choropleth
st.subheader("District Risk Map")
fig = px.choropleth(
    view,
    geojson=districts_geo,
    featureidkey="properties.district",
    locations="district",
    color=metric_choice,
    color_continuous_scale="Reds",
    hover_data=["total_enrolments", "update_rate", "p_failure", "impact", "risk_per_capita", "risk_level"],
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(height=600, margin={"r": 0, "t": 10, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

# Composite vs baseline comparison — the "does the model earn its complexity?" view
st.subheader("Composite risk vs naive baseline (1 − update_rate)")
left, right = st.columns(2)
with left:
    st.markdown(
        "If the composite were just a relabelling of `1 - update_rate`, this scatter "
        "would be a perfect diagonal. Spread off the diagonal = districts where the "
        "balance + impact components shift the ranking."
    )
    fig2 = px.scatter(
        view,
        x="baseline_risk",
        y="identity_maintenance_risk",
        color="risk_level",
        hover_data=["district", "total_enrolments"],
        trendline="ols",
    )
    fig2.update_layout(height=420, margin={"t": 10})
    st.plotly_chart(fig2, use_container_width=True)
with right:
    st.markdown("**Top-20 districts by composite risk**")
    st.dataframe(
        view.nlargest(20, "identity_maintenance_risk")[
            [
                "district",
                "total_enrolments",
                "update_rate",
                "p_failure",
                "impact",
                "identity_maintenance_risk",
            ]
        ].round(3),
        use_container_width=True,
        height=420,
    )

st.caption(
    "Limitations: index weights are configurable in `utils/config.py` and should be "
    "validated against ground-truth authentication-failure data before policy use. "
    "Demographic balance is a proxy and not a direct measure of exclusion."
)
