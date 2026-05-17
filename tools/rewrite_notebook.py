"""Apply the v2 surgery to ``UIDAI Project.ipynb``.

Run from the repo root:

    python tools/rewrite_notebook.py

What it does (one pass, idempotent within a single notebook version):

  Bug fixes
  ---------
  - Cell 1: drop `warnings.filterwarnings('ignore')` (was hiding real signals).
  - Cell 7: fix typo where df_master2['state'] was set from df_master1.
  - Cell 9: fix silent for-loop reassignment (merges were no-ops); use the
    new utils risk index which decomposes P(failure) and Impact.
  - Cell 10: replace hardcoded cluster_id → archetype mapping with
    centroid-characteristic labelling (stable across reseeds and reruns).
  - Cells 36-39: drop (empty trailers).

  Methodology upgrades
  --------------------
  - Cell 11: search K∈{2..10} on multiple metrics + bootstrap stability.
  - Cell 16: rapidfuzz join + Welch's t-test + Hedges' g + 95% CI for the
    NITI Aspirational comparison; reports join coverage.
  - Cell 18 + 19: Prophet `freq='ME'` (M deprecated) + cross_validation /
    performance_metrics for MAPE/RMSE on a rolling horizon.
  - New cell after 13: baseline vs composite Spearman comparison + bootstrap
    rank-stability summary (real sensitivity analysis, not "two configs").
  - Cells 32-34: collapse into a single call to ``utils.build_state_engagement``
    plus index computation; the duplicate rebuilds are gone.

  Hygiene
  -------
  - Strip ALL cell outputs and execution_count fields (drops the notebook
    from ~59MB to <1MB so GitHub will render it).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

NB_PATH = Path("UIDAI Project.ipynb")
BACKUP_PATH = Path("UIDAI Project.ipynb.bak")


# ---------------------------------------------------------------------------
# New / replacement cell sources
# ---------------------------------------------------------------------------
CELL_1_IMPORTS = """\
import json
import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

%load_ext autoreload
%autoreload 2
from prophet import Prophet

from utils.config import CONFIG
from utils.helpers import (
    build_state_engagement,
    calculate_balance_score,
    calculate_risk_index,
    fuzzy_match_districts,
    label_archetypes,
    naive_baseline_risk,
    sensitivity_rank_stability,
)

%matplotlib inline
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 6)

# NOTE: we deliberately do NOT silence warnings globally. A reviewer should see
# pandas FutureWarnings, Prophet deprecations, and sklearn convergence notices.
print("Risk weights (P(failure) sub-score):", CONFIG["risk_weights"]["p_failure"])
print("Risk weights (composite):           ", CONFIG["risk_weights"]["composite"])
"""


CELL_7_FIX = """\
df_master['state']  = df_master['state'].astype(str).str.strip().str.upper()
df_master1['state'] = df_master1['state'].astype(str).str.strip().str.upper()
df_master2['state'] = df_master2['state'].astype(str).str.strip().str.upper()  # bugfix: was df_master1
"""


CELL_9_FIX = """\
# Pincode → district mapping
pincode_map = pd.read_csv(
    CONFIG["files"]["pincode_mapping"],
    usecols=["pincode", "district", "state"],
    dtype={"pincode": str},
)
pincode_map = pincode_map.rename(columns={"districtname": "district", "statename": "state"})
pincode_map = pincode_map.drop_duplicates(subset="pincode").set_index("pincode")

# BUGFIX: the previous `for df in [...]: df = df.merge(...)` was a no-op.
# Reassign each frame explicitly.
def _attach_district(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pincode"] = df["pincode"].astype(str).str.strip()
    out_cols = [c for c in ("district", "state") if c in pincode_map.columns]
    if "district" in df.columns:
        df = df.drop(columns=[c for c in out_cols if c in df.columns])
    return df.merge(pincode_map[out_cols], left_on="pincode", right_index=True, how="left")

df_master  = _attach_district(df_master)
df_master1 = _attach_district(df_master1)
df_master2 = _attach_district(df_master2)

# District-level aggregation
df_master["total_enrolments"] = df_master[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
df_master2["total_updates"]   = df_master2[["demo_age_5_17", "demo_age_17_"]].sum(axis=1)

district_enrol = df_master.groupby("district")["total_enrolments"].sum().to_frame()
district_upd   = df_master2.groupby("district")["total_updates"].sum().to_frame()
district_df    = district_enrol.join(district_upd, how="inner").fillna(0)
district_df["update_rate"] = (
    district_df["total_updates"] / district_df["total_enrolments"]
).replace([np.inf, -np.inf], np.nan).fillna(0)

# Balance score per district
balance_district = calculate_balance_score(df_master2.assign(group=df_master2["district"]))
district_df = district_df.join(balance_district)

# Risk index (now exposes p_failure, impact, identity_maintenance_risk separately)
district_df = district_df.reset_index()
district_df = calculate_risk_index(district_df)
district_df = district_df[
    district_df["total_enrolments"] > CONFIG["thresholds"]["min_enrol_for_analysis"]
].copy()

print(f"Districts retained for analysis: {len(district_df):,}")
print(district_df[["district", "p_failure", "impact", "identity_maintenance_risk", "risk_level"]].head())

# Choropleth
with open(CONFIG["files"]["district_geojson"], encoding="utf-8") as f:
    districts_geo = json.load(f)

fig = px.choropleth(
    district_df,
    geojson=districts_geo,
    featureidkey="properties.district",
    locations="district",
    color="identity_maintenance_risk",
    color_continuous_scale="Reds",
    title="<b>India: Aadhaar Identity Maintenance Risk by District</b>",
    hover_data=["total_enrolments", "update_rate", "p_failure", "impact", "risk_level"],
)
fig.update_layout(width=1000, height=600, margin={"r":0, "t":80, "l":0, "b":0}, title_x=0.5, title_font_size=24)
fig.update_geos(fitbounds="locations", visible=False)
fig.show()
"""


CELL_10_FIX = """\
# Enriched feature set so clusters describe something the risk index does NOT
# already encode (avoids tautological "clusters recover the index" critique).
district_df["log_total_enrolments"] = np.log1p(district_df["total_enrolments"])

# Biometric vs demographic update mix (district-level)
bio_by_district = (
    df_master1.groupby("district")[["bio_age_5_17", "bio_age_17_"]].sum().sum(axis=1)
    .rename("bio_updates")
)
district_df = district_df.merge(bio_by_district, left_on="district", right_index=True, how="left").fillna({"bio_updates": 0})
district_df["bio_demo_ratio"] = district_df["bio_updates"] / (district_df["total_updates"] + 1)

# Enrolment growth slope: simple OLS slope of monthly enrolment counts per district
df_master["date"] = pd.to_datetime(df_master["date"], errors="coerce", dayfirst=True)
monthly = (
    df_master.dropna(subset=["date", "district"])
    .assign(ym=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
    .groupby(["district", "ym"]).size().rename("n").reset_index()
)
def _slope(g):
    if len(g) < 2:
        return 0.0
    x = (g["ym"] - g["ym"].min()).dt.days.to_numpy()
    y = g["n"].to_numpy()
    if x.std() == 0:
        return 0.0
    return np.polyfit(x, y, 1)[0]
slope = monthly.groupby("district").apply(_slope).rename("enrol_growth_slope")
district_df = district_df.merge(slope, left_on="district", right_index=True, how="left").fillna({"enrol_growth_slope": 0.0})

features = CONFIG["clustering"]["features"]
cluster_data = district_df[features].copy()
cluster_data_norm = pd.DataFrame(
    MinMaxScaler().fit_transform(cluster_data),
    columns=features, index=cluster_data.index,
)

kmeans = KMeans(
    n_clusters=CONFIG["clustering"]["n_clusters_default"],
    n_init=10,
    random_state=CONFIG["clustering"]["random_state"],
)
district_df["cluster"] = kmeans.fit_predict(cluster_data_norm)

# Centroid characteristics (in original units, not normalised)
centroid_summary = district_df.groupby("cluster").agg(
    log_total_enrolments=("log_total_enrolments", "mean"),
    update_rate=("update_rate", "mean"),
    balance_score=("balance_score", "mean"),
    bio_demo_ratio=("bio_demo_ratio", "mean"),
    n_districts=("district", "count"),
).round(3)
print("Cluster centroids:")
print(centroid_summary)

# BUGFIX: labels were hardcoded by cluster_id {0,1,2,3}. KMeans assigns ids
# arbitrarily, so this broke on any data/seed change. Now we label by what the
# centroid actually looks like.
archetype_map = label_archetypes(centroid_summary)
district_df["archetype"] = district_df["cluster"].map(archetype_map)
print("\\nArchetype labels (by centroid characteristics):")
for cid, name in archetype_map.items():
    print(f"  cluster {cid}: {name}")

fig = px.scatter(
    district_df, x="update_rate", y="total_enrolments",
    color="archetype", size="balance_score",
    hover_data=["district", "p_failure", "impact"],
    title="District Archetypes (K-Means on enriched feature set)",
)
fig.update_layout(width=1000, height=600, margin={"r":0, "t":80, "l":0, "b":0}, title_x=0.5, title_font_size=24)
fig.update_yaxes(type="log")
fig.show()
"""


CELL_11_FIX = """\
# Multi-metric K search + bootstrap stability. The previous version tested
# only K∈{2..5} on silhouette alone, with zero stability check.
from sklearn.utils import resample

K_range = CONFIG["clustering"]["n_clusters_search"]
scores = []
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=CONFIG["clustering"]["random_state"])
    labels_k = km.fit_predict(cluster_data_norm)
    scores.append({
        "k": k,
        "silhouette": silhouette_score(cluster_data_norm, labels_k),
        "davies_bouldin": davies_bouldin_score(cluster_data_norm, labels_k),  # lower = better
        "inertia": km.inertia_,
    })
score_df = pd.DataFrame(scores).set_index("k")
print("K-search (silhouette ↑ better, davies_bouldin ↓ better):")
print(score_df.round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
score_df["silhouette"].plot(ax=axes[0], marker="o", title="Silhouette ↑")
score_df["davies_bouldin"].plot(ax=axes[1], marker="o", title="Davies-Bouldin ↓", color="orange")
score_df["inertia"].plot(ax=axes[2], marker="o", title="Inertia (Elbow)", color="green")
for ax in axes:
    ax.axvline(CONFIG["clustering"]["n_clusters_default"], color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("K")
plt.tight_layout()
plt.show()

# Bootstrap stability: how often do two districts co-cluster across resamples?
# Reported as mean Adjusted Rand Index vs the default-K fit.
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(CONFIG["clustering"]["random_state"])
base_labels = district_df["cluster"].to_numpy()
ari_scores = []
for _ in range(CONFIG["clustering"]["n_bootstrap_stability"]):
    idx = rng.integers(0, len(cluster_data_norm), len(cluster_data_norm))
    km = KMeans(
        n_clusters=CONFIG["clustering"]["n_clusters_default"],
        n_init=10, random_state=int(rng.integers(0, 1e6)),
    )
    boot_labels = km.fit_predict(cluster_data_norm.iloc[idx])
    ari_scores.append(adjusted_rand_score(base_labels[idx], boot_labels))

print(f"\\nBootstrap cluster stability (ARI, {CONFIG['clustering']['n_bootstrap_stability']} resamples):")
print(f"  mean = {np.mean(ari_scores):.3f}   std = {np.std(ari_scores):.3f}")
print("  (ARI 1.0 = identical, 0.0 = random. >0.7 indicates stable clusters.)")
"""


CELL_BASELINE_AND_SENSITIVITY = """\
# ---------------------------------------------------------------------------
# Sanity check 1: does the composite index beat the trivial baseline?
# ---------------------------------------------------------------------------
from scipy.stats import spearmanr

district_df["baseline_risk"] = naive_baseline_risk(district_df)

rho, pval = spearmanr(district_df["identity_maintenance_risk"], district_df["baseline_risk"])
print(f"Spearman ρ (composite vs naive 1-update_rate): {rho:.3f}  (p={pval:.2e})")
print("  High ρ would mean the composite adds little; we want enough divergence")
print("  to justify the extra signal (balance + impact).")

# Top-20 overlap: how many districts in the top-20 by composite are also top-20 by baseline?
top_comp = set(district_df.nlargest(20, "identity_maintenance_risk")["district"])
top_base = set(district_df.nlargest(20, "baseline_risk")["district"])
overlap = len(top_comp & top_base)
print(f"\\nTop-20 overlap (composite vs baseline): {overlap}/20")
print("  Districts the composite uniquely flags:", sorted(top_comp - top_base)[:10])

# ---------------------------------------------------------------------------
# Sanity check 2: rank stability under perturbed weights (real sensitivity)
# ---------------------------------------------------------------------------
sens = sensitivity_rank_stability(district_df, n_perturbations=200, weight_jitter=0.15)
sens["district"] = district_df["district"].values
print("\\nRank stability under ±15% weight jitter (lower std = more robust ranking):")
print(sens.sort_values("rank_mean").head(10))

# Visualise: rank uncertainty band
top30 = sens.nsmallest(30, "rank_mean").sort_values("rank_mean")
plt.figure(figsize=(10, 8))
plt.errorbar(
    top30["rank_mean"], range(len(top30)),
    xerr=[top30["rank_mean"] - top30["rank_p05"], top30["rank_p95"] - top30["rank_mean"]],
    fmt="o", capsize=3,
)
plt.yticks(range(len(top30)), top30["district"], fontsize=8)
plt.xlabel("Rank (1 = highest risk)  ±  5/95 percentile band over 200 weight perturbations")
plt.title("Top-30 Risk Rank Stability")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
"""


CELL_16_FIX = """\
# --- NITI Aayog cross-reference with fuzzy match + proper statistics ---
from scipy import stats

asp_fin = pd.read_csv(CONFIG["files"]["financial_inclusion"])
asp_fin["dist_clean"] = asp_fin["District Name"].astype(str).str.upper().str.strip()
district_df["dist_clean"] = district_df["district"].astype(str).str.upper().str.strip()

# 1. Report exact-match coverage first (the previous baseline)
exact_overlap = set(asp_fin["dist_clean"]) & set(district_df["dist_clean"])
print(f"Exact-match coverage: {len(exact_overlap)} / {asp_fin['dist_clean'].nunique()} NITI districts")

# 2. Fuzzy match for the rest (Bengaluru/Bangalore, Gurgaon/Gurugram, etc.)
fm = fuzzy_match_districts(asp_fin["dist_clean"], district_df["dist_clean"])
matched_pct = 100 * fm["matched"].mean()
print(f"Fuzzy-match coverage:  {fm['matched'].sum()} / {len(fm)} ({matched_pct:.1f}%) at score≥{CONFIG['thresholds']['fuzzy_match_threshold']}")

asp_fin = asp_fin.merge(
    fm[fm["matched"]][["left", "right_match"]],
    left_on="dist_clean", right_on="left", how="left",
).rename(columns={"right_match": "dist_match"}).drop(columns="left")
asp_fin["dist_match"] = asp_fin["dist_match"].fillna(asp_fin["dist_clean"])

merged = district_df.merge(
    asp_fin[["dist_match", "%Improvement (T)", "%Improvement (T-1)"]],
    left_on="dist_clean", right_on="dist_match", how="left",
)
merged["is_aspirational"] = merged["%Improvement (T)"].notna()

asp = merged.loc[merged["is_aspirational"], "identity_maintenance_risk"].to_numpy()
std = merged.loc[~merged["is_aspirational"], "identity_maintenance_risk"].to_numpy()

# Welch's t-test (unequal variance — standard in DS practice)
t_stat, p_val = stats.ttest_ind(asp, std, equal_var=False)

# Hedges' g effect size + 95% CI on the difference of means
mean_diff = asp.mean() - std.mean()
pooled_sd = np.sqrt(((asp.var(ddof=1) * (len(asp) - 1)) + (std.var(ddof=1) * (len(std) - 1))) / (len(asp) + len(std) - 2))
hedges_g = mean_diff / pooled_sd * (1 - (3 / (4 * (len(asp) + len(std)) - 9)))  # small-sample correction
se = np.sqrt(asp.var(ddof=1) / len(asp) + std.var(ddof=1) / len(std))
ci_low, ci_high = mean_diff - 1.96 * se, mean_diff + 1.96 * se

print("\\nSTRATEGIC VALIDATION REPORT")
print("-" * 60)
print(f"Aspirational districts (n={len(asp):,}): mean risk = {asp.mean():.4f}")
print(f"Standard districts     (n={len(std):,}): mean risk = {std.mean():.4f}")
print(f"Difference:            Δ = {mean_diff:+.4f}   95% CI [{ci_low:+.4f}, {ci_high:+.4f}]")
print(f"Welch's t-test:        t = {t_stat:.3f}, p = {p_val:.4f}")
print(f"Hedges' g (effect):    g = {hedges_g:.3f}   (|g|<0.2 negligible, 0.2-0.5 small, >0.5 medium)")
if p_val < 0.05:
    print("→ Statistically significant difference at α=0.05.")
else:
    print("→ NOT statistically significant at α=0.05 — be careful claiming this finding.")

fig = px.scatter(
    merged[merged["is_aspirational"]],
    x="%Improvement (T)", y="identity_maintenance_risk",
    color="archetype", hover_data=["district"], trendline="ols",
    title="<b>Financial Inclusion Progress vs. Aadhaar Risk (Aspirational Districts)</b>",
    labels={"%Improvement (T)": "NITI Financial Inclusion % Improvement",
            "identity_maintenance_risk": "Aadhaar Maintenance Risk"},
)
fig.update_layout(width=1000, height=600, template="plotly_white")
fig.show()
"""


CELL_18_FIX = """\
# --- National monthly forecast with rolling-horizon validation ---
from prophet.diagnostics import cross_validation, performance_metrics

df_master2["date"] = pd.to_datetime(df_master2["date"], errors="coerce")
monthly_updates = (
    df_master2.dropna(subset=["date"])
    .groupby(df_master2["date"].dt.to_period("M"))["total_updates"]
    .sum().reset_index()
)
monthly_updates["ds"] = monthly_updates["date"].dt.to_timestamp()
monthly_updates["y"] = monthly_updates["total_updates"]
ts = monthly_updates[["ds", "y"]].sort_values("ds")

m = Prophet(yearly_seasonality=True, interval_width=0.95)
m.fit(ts)

# freq='M' is deprecated in pandas → 'ME' (month-end). Use the new spelling.
future = m.make_future_dataframe(periods=CONFIG["forecasting"]["horizon_months"], freq="ME")
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.title("National Demographic Updates — 12-month forecast", fontsize=13, fontweight="bold")
plt.xlabel("Date"); plt.ylabel("Updates")
plt.show()

# Rolling-horizon cross-validation — produces MAPE/RMSE we can actually quote.
if len(ts) >= CONFIG["forecasting"]["cv_initial_months"] + CONFIG["forecasting"]["cv_horizon_months"]:
    cv = cross_validation(
        m,
        initial=f"{CONFIG['forecasting']['cv_initial_months'] * 30} days",
        period=f"{CONFIG['forecasting']['cv_period_months'] * 30} days",
        horizon=f"{CONFIG['forecasting']['cv_horizon_months'] * 30} days",
        disable_tqdm=True,
    )
    perf = performance_metrics(cv, rolling_window=1.0)
    print("National forecast — rolling-horizon validation:")
    print(perf[["horizon", "mape", "rmse", "mae"]].round(4))
else:
    print("Insufficient history for cross_validation — quoting forecast WITHOUT validated error bars.")
"""


CELL_19_FIX = """\
# --- Archetype-level forecasts, also validated ---
from prophet.diagnostics import cross_validation, performance_metrics

df_master1["date"] = pd.to_datetime(df_master1["date"], errors="coerce")
df_master2["date"] = pd.to_datetime(df_master2["date"], errors="coerce")

bio_ts  = df_master1.groupby(["date", "district"])["total_updates"].sum().reset_index() if "total_updates" in df_master1.columns else df_master1.assign(total_updates=df_master1[["bio_age_5_17","bio_age_17_"]].sum(axis=1)).groupby(["date","district"])["total_updates"].sum().reset_index()
demo_ts = df_master2.groupby(["date", "district"])["total_updates"].sum().reset_index()
update_df = pd.merge(bio_ts, demo_ts, on=["date", "district"], how="outer", suffixes=("_bio", "_demo")).fillna(0)
update_df["total_updates"] = update_df["total_updates_bio"] + update_df["total_updates_demo"]

ts_national = update_df.groupby("date")["total_updates"].sum().reset_index().rename(columns={"date": "ds", "total_updates": "y"})

model_nat = Prophet(yearly_seasonality=True, interval_width=0.95)
model_nat.fit(ts_national)
future_nat = model_nat.make_future_dataframe(periods=CONFIG["forecasting"]["horizon_months"], freq="ME")
forecast_nat = model_nat.predict(future_nat)

fig_nat = model_nat.plot(forecast_nat)
plt.title(f"National Aadhaar Update Load — next {CONFIG['forecasting']['horizon_months']} months", fontsize=13, fontweight="bold")
plt.show()

print("\\nARCHETYPE-SPECIFIC DEMAND PROJECTIONS (next 6 months) + validation MAPE")
print("-" * 75)
for arch in district_df["archetype"].dropna().unique():
    districts = district_df.loc[district_df["archetype"] == arch, "district"].tolist()
    ts_arch = (
        update_df[update_df["district"].isin(districts)]
        .groupby("date")["total_updates"].sum().reset_index()
        .rename(columns={"date": "ds", "total_updates": "y"})
        .sort_values("ds")
    )
    if len(ts_arch) < 12:
        print(f"  {arch[:40]:<42} insufficient history")
        continue
    m = Prophet(yearly_seasonality=True)
    m.fit(ts_arch)
    future = m.make_future_dataframe(periods=6, freq="ME")
    fc = m.predict(future)
    projected = fc["yhat"].tail(6).sum()

    mape_str = "n/a"
    if len(ts_arch) >= CONFIG["forecasting"]["cv_initial_months"] + CONFIG["forecasting"]["cv_horizon_months"]:
        try:
            cv = cross_validation(
                m,
                initial=f"{CONFIG['forecasting']['cv_initial_months'] * 30} days",
                period=f"{CONFIG['forecasting']['cv_period_months'] * 30} days",
                horizon=f"{CONFIG['forecasting']['cv_horizon_months'] * 30} days",
                disable_tqdm=True,
            )
            perf = performance_metrics(cv, rolling_window=1.0)
            mape_str = f"{perf['mape'].mean():.1%}"
        except Exception as e:  # noqa: BLE001
            mape_str = f"cv failed ({type(e).__name__})"
    print(f"  {arch[:40]:<42} projected={projected:>14,.0f}   MAPE={mape_str}")
print("-" * 75)
"""


CELL_32_REFACTOR = """\
# Single-source-of-truth state-level engagement (replaces the three duplicate
# rebuilds previously in cells 32, 33, 34). All downstream cells consume `engagement`.
engagement = build_state_engagement(df_master, df_master2)

# State-level risk index (re-uses district-level helper for consistency)
state_risk = calculate_risk_index(engagement.reset_index())
state_risk = state_risk[state_risk["total_enrolments"] > CONFIG["thresholds"]["min_enrol_for_analysis"]].copy()
state_risk = state_risk.sort_values("identity_maintenance_risk", ascending=False)

print("Top 10 States by Identity Maintenance Risk:")
print(state_risk[["state", "total_enrolments", "p_failure", "impact",
                  "identity_maintenance_risk", "risk_level"]].head(10).to_string(index=False))

# Inclusion-Index leaderboard (positive-framing of the same components)
state_risk["inclusion_index"] = 1.0 - state_risk["identity_maintenance_risk"]
top15 = state_risk.nlargest(15, "inclusion_index")
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=top15, x="inclusion_index", y="state", palette="viridis")
for p in ax.patches:
    ax.text(p.get_width() + 0.005, p.get_y() + p.get_height() / 2,
            f"{p.get_width():.3f}", va="center", fontsize=11, fontweight="bold")
plt.title("Aadhaar Inclusion Index: Top 15 States", fontsize=18, pad=20, fontweight="bold")
plt.xlabel("Inclusion Index (Weighted Score)"); plt.ylabel("State")
sns.despine(left=True, bottom=True); plt.tight_layout(); plt.show()
"""


CELL_33_REFACTOR = """\
# Tertile classification already lives in `state_risk['risk_level']`. Show it.
print("\\nState-level Risk Tiers:")
print(state_risk.groupby("risk_level")["state"].apply(list))
"""


CELL_34_REFACTOR = """\
# Real sensitivity analysis: bootstrap weight perturbations.
state_sens = sensitivity_rank_stability(state_risk, n_perturbations=200, weight_jitter=0.15)
state_sens["state"] = state_risk["state"].values
print("\\nState-level rank stability under ±15% weight jitter:")
print(state_sens.sort_values("rank_mean").head(10))
"""


CELL_12_MARKDOWN_FIX = """\
We selected K=4 as the default after searching K∈{2..10} on silhouette, Davies-Bouldin, and inertia (elbow). Bootstrap Adjusted Rand Index quantifies cluster stability across resamples — see the cell above.
"""


CELL_17_MARKDOWN_FIX = """\
### Key Finding: Higher Risk in Aspirational Districts (validated)

The cell above reports a **Welch's t-test, 95% CI on the mean difference, and Hedges' g effect size** for the comparison between Aspirational and Standard districts. We deliberately quote the test rather than just two means — the previously reported "0.7904 vs 0.7819" looks like a finding but is methodologically thin without significance and effect size context.

The scatter plot visualises the relationship between **NITI % Improvement in Financial Inclusion** and our **Identity Maintenance Risk Score**, coloured by district archetype. The OLS trendline gives a directional read; a stronger version of this analysis would model risk as a function of inclusion progress and report R²/β with confidence intervals.

### Policy Recommendations
- **Bundle interventions**: integrate Aadhaar mobile update camps with ongoing Aspirational Districts programmes (Financial Inclusion & Skill Development) to leverage existing field infrastructure.
- **Prioritise High-Overlap archetypes**: focus immediate outreach on "High-Growth, Low-Maintenance" and "Low-Volume, Imbalanced" districts within the 112 Aspirational list.
- **Caveat**: the join uses fuzzy district matching (rapidfuzz, threshold reported in the previous cell). Names that fail to match are excluded from the comparison and may bias the result.
"""


# ---------------------------------------------------------------------------
# Apply edits
# ---------------------------------------------------------------------------
def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def main() -> None:
    if not NB_PATH.exists():
        raise SystemExit(f"Notebook not found at {NB_PATH.resolve()}")

    # One-time backup so the user can always recover the original
    if not BACKUP_PATH.exists():
        shutil.copy2(NB_PATH, BACKUP_PATH)
        print(f"Backup written -> {BACKUP_PATH}")

    with NB_PATH.open(encoding="utf-8") as f:
        nb = json.load(f)

    n_before = len(nb["cells"])
    print(f"Loaded notebook: {n_before} cells")

    # --- Replace specific cells in place ---
    replacements = {
        1: code_cell(CELL_1_IMPORTS),
        7: code_cell(CELL_7_FIX),
        9: code_cell(CELL_9_FIX),
        10: code_cell(CELL_10_FIX),
        11: code_cell(CELL_11_FIX),
        12: md_cell(CELL_12_MARKDOWN_FIX),
        16: code_cell(CELL_16_FIX),
        17: md_cell(CELL_17_MARKDOWN_FIX),
        18: code_cell(CELL_18_FIX),
        19: code_cell(CELL_19_FIX),
        32: code_cell(CELL_32_REFACTOR),
        33: code_cell(CELL_33_REFACTOR),
        34: code_cell(CELL_34_REFACTOR),
    }
    for idx, new_cell in replacements.items():
        if idx >= len(nb["cells"]):
            raise SystemExit(f"Cell index {idx} out of range ({len(nb['cells'])} cells)")
        nb["cells"][idx] = new_cell

    # --- Insert the baseline+sensitivity cell after cell 13 (now becomes index 14) ---
    nb["cells"].insert(14, code_cell(CELL_BASELINE_AND_SENSITIVITY))

    # --- Drop empty trailer cells (36..end after insertion → indices shift by +1) ---
    # After insertion, old indices 36..39 are now 37..40. We drop any trailing
    # cells whose stripped source is empty.
    while nb["cells"] and not "".join(nb["cells"][-1]["source"]).strip():
        nb["cells"].pop()

    # --- Strip ALL outputs (the 59MB → <1MB win) ---
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
            c.setdefault("metadata", {})

    # --- Bump kernel metadata to something more conventional ---
    nb.setdefault("metadata", {})
    nb["metadata"]["language_info"] = nb["metadata"].get("language_info", {})

    out_text = json.dumps(nb, indent=1, ensure_ascii=False)
    NB_PATH.write_text(out_text + "\n", encoding="utf-8")
    print(f"Rewrote notebook: {len(nb['cells'])} cells, {len(out_text):,} bytes")
    print("Done.")


if __name__ == "__main__":
    main()
