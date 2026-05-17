"""Reproduce just the headline numbers we need for the README, without the
plotly figures that bloat the executed notebook.

Run from repo root:
    python tools/headline_numbers.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

from utils.config import CONFIG
from utils.helpers import (
    calculate_balance_score,
    calculate_risk_index,
    fuzzy_match_districts,
    naive_baseline_risk,
    risk_per_capita,
    temporal_holdout_rank_stability,
)


def load_concat(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def main() -> None:
    print("\n=== Loading raw data ===")
    df_master = clean(load_concat("datasets/api_data_aadhar_enrolment"))
    df_master1 = clean(load_concat("datasets/api_data_aadhar_biometric"))
    df_master2 = clean(load_concat("datasets/api_data_aadhar_demographic"))
    for df in (df_master, df_master1, df_master2):
        df["pincode"] = df["pincode"].astype(str).str.strip()
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.strip().str.upper()
    print(f"  enrol: {len(df_master):,}  bio: {len(df_master1):,}  demo: {len(df_master2):,}")

    pincode_map = (
        pd.read_csv(
            CONFIG["files"]["pincode_mapping"],
            usecols=["pincode", "district", "state"],
            dtype={"pincode": str},
        )
        .drop_duplicates(subset="pincode")
        .set_index("pincode")
    )

    def attach(df):
        df = df.copy()
        out_cols = [c for c in ("district", "state") if c in pincode_map.columns]
        df = df.drop(columns=[c for c in out_cols if c in df.columns])
        return df.merge(pincode_map[out_cols], left_on="pincode", right_index=True, how="left")

    df_master = attach(df_master)
    df_master1 = attach(df_master1)
    df_master2 = attach(df_master2)

    df_master["total_enrolments"] = df_master[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
    df_master2["total_updates"] = df_master2[["demo_age_5_17", "demo_age_17_"]].sum(axis=1)

    district_df = (
        df_master.groupby("district")["total_enrolments"]
        .sum()
        .to_frame()
        .join(df_master2.groupby("district")["total_updates"].sum().to_frame(), how="inner")
        .fillna(0)
    )
    district_df["update_rate"] = (
        (district_df["total_updates"] / district_df["total_enrolments"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    bal = calculate_balance_score(df_master2.assign(group=df_master2["district"]))
    district_df = district_df.join(bal).reset_index()
    district_df = calculate_risk_index(district_df)
    district_df = district_df[
        district_df["total_enrolments"] > CONFIG["thresholds"]["min_enrol_for_analysis"]
    ].copy()
    district_df["baseline_risk"] = naive_baseline_risk(district_df)
    print(f"  districts analysed: {len(district_df):,}")

    # --------- Sanity check: composite vs baseline ---------
    print("\n=== Composite vs naive baseline ===")
    rho, pval = spearmanr(district_df["identity_maintenance_risk"], district_df["baseline_risk"])
    top_comp = set(district_df.nlargest(20, "identity_maintenance_risk")["district"])
    top_base = set(district_df.nlargest(20, "baseline_risk")["district"])
    print(f"  Spearman rho = {rho:.3f}  (p = {pval:.2e})")
    print(f"  Top-20 overlap = {len(top_comp & top_base)}/20")

    # --------- Per-capita comparison ---------
    print("\n=== Per-capita view (strip Impact term) ===")
    district_df["risk_per_capita"] = risk_per_capita(district_df).values
    top_pc = set(district_df.nlargest(20, "risk_per_capita")["district"])
    print(f"  Top-20 overlap (composite vs per-capita) = {len(top_comp & top_pc)}/20")

    # --------- Bias slice: quartile trend ---------
    print("\n=== Bias slice: risk by enrolment quartile ===")
    district_df["enrol_quartile"] = pd.qcut(
        district_df["total_enrolments"],
        q=4,
        labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"],
    )
    print(
        district_df.groupby("enrol_quartile", observed=True)["identity_maintenance_risk"]
        .median()
        .round(4)
        .to_string()
    )

    # --------- NITI unadjusted + stratified ---------
    print("\n=== NITI Aspirational comparison ===")
    asp_fin = pd.read_csv(CONFIG["files"]["financial_inclusion"])
    asp_fin["dist_clean"] = asp_fin["District Name"].astype(str).str.upper().str.strip()
    district_df["dist_clean"] = district_df["district"].astype(str).str.upper().str.strip()
    fm = fuzzy_match_districts(asp_fin["dist_clean"], district_df["dist_clean"])
    asp_fin = (
        asp_fin.merge(
            fm[fm["matched"]][["left", "right_match"]],
            left_on="dist_clean",
            right_on="left",
            how="left",
        )
        .rename(columns={"right_match": "dist_match"})
        .drop(columns="left")
    )
    asp_fin["dist_match"] = asp_fin["dist_match"].fillna(asp_fin["dist_clean"])

    merged = district_df.merge(
        asp_fin[["dist_match", "%Improvement (T)", "%Improvement (T-1)"]],
        left_on="dist_clean",
        right_on="dist_match",
        how="left",
    )
    merged["is_aspirational"] = merged["%Improvement (T)"].notna()

    asp = merged.loc[merged["is_aspirational"], "identity_maintenance_risk"].to_numpy()
    std = merged.loc[~merged["is_aspirational"], "identity_maintenance_risk"].to_numpy()
    t_stat, p_val = stats.ttest_ind(asp, std, equal_var=False)
    mean_diff = asp.mean() - std.mean()
    pooled_sd = np.sqrt(
        ((asp.var(ddof=1) * (len(asp) - 1)) + (std.var(ddof=1) * (len(std) - 1))) / (len(asp) + len(std) - 2)
    )
    hedges_g = mean_diff / pooled_sd * (1 - (3 / (4 * (len(asp) + len(std)) - 9)))
    se = np.sqrt(asp.var(ddof=1) / len(asp) + std.var(ddof=1) / len(std))
    ci_low, ci_high = mean_diff - 1.96 * se, mean_diff + 1.96 * se
    print(f"  Unadjusted: n_asp={len(asp)} n_std={len(std)}")
    print(f"    diff = {mean_diff:+.4f}  95% CI [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"    t = {t_stat:.3f}, p = {p_val:.4f}, Hedges g = {hedges_g:.3f}")

    # Attach state from df_master via mode lookup
    dist_to_state = (
        df_master.dropna(subset=["district", "state"])
        .groupby("district")["state"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
    )
    merged = merged.merge(dist_to_state.rename("state"), left_on="district", right_index=True, how="left")
    merged["risk_demeaned"] = merged.groupby("state")["identity_maintenance_risk"].transform(
        lambda x: x - x.mean()
    )
    asp_dm = merged.loc[merged["is_aspirational"], "risk_demeaned"].dropna().to_numpy()
    std_dm = merged.loc[~merged["is_aspirational"], "risk_demeaned"].dropna().to_numpy()
    t_dm, p_dm = stats.ttest_ind(asp_dm, std_dm, equal_var=False)
    diff_dm = asp_dm.mean() - std_dm.mean()
    print(f"  Within-state stratified: n_asp={len(asp_dm)} n_std={len(std_dm)}")
    print(f"    demeaned diff = {diff_dm:+.4f}  t = {t_dm:.3f}, p = {p_dm:.4f}")

    # --------- Temporal hold-out ---------
    print("\n=== Temporal hold-out (last 3 months held out) ===")
    holdout = temporal_holdout_rank_stability(
        df_enrol=df_master,
        df_demo=df_master2,
        date_col="date",
        group_col="district",
        holdout_months=3,
    )
    for k, v in holdout.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
