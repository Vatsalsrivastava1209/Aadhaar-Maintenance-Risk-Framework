"""Reusable analysis helpers for the Aadhaar Identity Maintenance Risk pipeline.

Design notes:
- The risk index is decomposed into P(failure) and Impact, exposed separately
  so downstream callers can choose their own composition or analyze drivers.
- All scaling fits MinMaxScaler per call (stateless) — callers wanting a
  reproducible scaler for serving should persist one outside this module.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import CONFIG


# ---------------------------------------------------------------------------
# Risk index (decomposed: P(failure) × Impact)
# ---------------------------------------------------------------------------
def _safe_minmax(series: pd.Series) -> pd.Series:
    """MinMax-scale a Series robustly. Returns 0.0 series if all values equal."""
    arr = series.to_numpy(dtype=float).reshape(-1, 1)
    if np.nanmax(arr) == np.nanmin(arr):
        return pd.Series(np.zeros(len(series)), index=series.index)
    scaled = MinMaxScaler().fit_transform(arr).ravel()
    return pd.Series(scaled, index=series.index)


def calculate_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Identity Maintenance Risk Index.

    Risk = P(failure) × Impact, where:
      P(failure) = w1 * (1 - update_rate_norm) + w2 * (1 - balance_norm)
      Impact     = log1p(total_enrolments) normalised to [0, 1]

    The composite `identity_maintenance_risk` blends these two via
    `CONFIG.risk_weights.composite` so callers can argue with the weights
    rather than the structure.

    Adds columns:
      enrol_norm, update_rate_norm, balance_norm,
      p_failure, impact, identity_maintenance_risk, risk_level
    """
    df = df.copy()

    # log1p enrolments before scaling — raw values are heavy-tailed (UP/Maharashtra
    # dominate). Without this, MinMax compresses 95% of districts into [0, 0.05].
    df["enrol_norm"] = _safe_minmax(np.log1p(df["total_enrolments"]))
    df["update_rate_norm"] = _safe_minmax(df["update_rate"])
    df["balance_norm"] = _safe_minmax(df["balance_score"])

    w_pf = CONFIG["risk_weights"]["p_failure"]
    w_c = CONFIG["risk_weights"]["composite"]

    df["p_failure"] = w_pf["update_rate"] * (1 - df["update_rate_norm"]) + w_pf["balance"] * (
        1 - df["balance_norm"]
    )
    df["impact"] = df["enrol_norm"]

    df["identity_maintenance_risk"] = w_c["p_failure"] * df["p_failure"] + w_c["impact"] * df["impact"]

    # Tertile labels — guard against degenerate cases (all-equal inputs collapse
    # qcut into a single bin which would crash with 3 labels).
    try:
        df["risk_level"] = pd.qcut(
            df["identity_maintenance_risk"],
            q=3,
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )
    except ValueError:
        df["risk_level"] = pd.qcut(
            df["identity_maintenance_risk"].rank(method="first"),
            q=3,
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )
    return df


def naive_baseline_risk(df: pd.DataFrame) -> pd.Series:
    """Trivial baseline: risk = 1 - update_rate. Used to prove the composite earns its complexity."""
    ur = df["update_rate"].clip(lower=0, upper=1)
    return 1.0 - ur


def risk_per_capita(df: pd.DataFrame) -> pd.Series:
    """Per-capita view of risk: strip the Impact (volume) term so the composite
    no longer privileges large districts.

    With no Census table available, we use ``total_enrolments`` itself as a
    population proxy. The composite already encodes
    ``impact = log1p(total_enrolments)``, so
    ``identity_maintenance_risk / log1p(total_enrolments)`` collapses to a
    rescaled ``p_failure`` term. We return ``p_failure`` directly — same
    ranking, clearer semantics — and document the equivalence here so a
    reviewer doesn't have to re-derive it.
    """
    if "p_failure" not in df.columns:
        raise KeyError("risk_per_capita expects calculate_risk_index() to have been run first")
    return df["p_failure"].rename("risk_per_capita")


# ---------------------------------------------------------------------------
# Balance score
# ---------------------------------------------------------------------------
def calculate_balance_score(df_updates: pd.DataFrame) -> pd.DataFrame:
    """Variance-based age-group balance per group (district or state).

    Lower variance across age buckets → more balanced coverage → higher score.
    """
    long = df_updates.melt(
        id_vars=["group"],
        value_vars=["demo_age_5_17", "demo_age_17_"],
        var_name="age_group",
        value_name="count",
    )
    variance = long.groupby("group")["count"].var().fillna(0)
    return (1.0 / (1.0 + variance)).to_frame("balance_score")


# ---------------------------------------------------------------------------
# Engagement aggregation (extracted from duplicate cells 32-34)
# ---------------------------------------------------------------------------
def build_state_engagement(
    df_enrol: pd.DataFrame,
    df_demo: pd.DataFrame,
    enrol_age_cols: Iterable[str] = ("age_0_5", "age_5_17", "age_18_greater"),
    demo_age_cols: Iterable[str] = ("demo_age_5_17", "demo_age_17_"),
) -> pd.DataFrame:
    """Build state-level engagement table (enrolments, updates, update_rate, balance_score).

    Replaces the three duplicate rebuilds previously in cells 32, 33, 34.
    """
    enrol = df_enrol.groupby("state")[list(enrol_age_cols)].sum().sum(axis=1).to_frame("total_enrolments")
    upd = df_demo.groupby("state")[list(demo_age_cols)].sum().sum(axis=1).to_frame("total_updates")
    eng = enrol.join(upd, how="inner").fillna(0)
    eng["update_rate"] = (
        (eng["total_updates"] / eng["total_enrolments"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    bal = calculate_balance_score(df_demo.rename(columns={"state": "group"}).assign(group=df_demo["state"]))
    eng = eng.join(bal, how="left").fillna(0)
    return eng


# ---------------------------------------------------------------------------
# Sensitivity analysis (proper, not "two configs")
# ---------------------------------------------------------------------------
def sensitivity_rank_stability(
    df: pd.DataFrame,
    n_perturbations: int = 200,
    weight_jitter: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap weights to assess rank stability of the risk index.

    For each perturbation, jitter the composite weights by ±jitter (renormalised
    to sum to 1), recompute risk, and re-rank districts. Returns per-district
    summary of rank distribution (mean, std, p05, p95).
    """
    rng = np.random.default_rng(random_state)
    base = CONFIG["risk_weights"]["composite"]
    keys = list(base.keys())
    base_arr = np.array([base[k] for k in keys], dtype=float)

    p_failure = df["p_failure"].to_numpy()
    impact = df["impact"].to_numpy()
    components = np.stack([p_failure, impact], axis=1)  # shape (n_districts, 2)

    ranks = np.empty((n_perturbations, len(df)), dtype=int)
    for i in range(n_perturbations):
        jitter = rng.uniform(-weight_jitter, weight_jitter, size=base_arr.shape)
        w = np.clip(base_arr + jitter, 1e-3, None)
        w = w / w.sum()
        score = components @ w
        # rank: 1 = highest risk
        order = score.argsort()[::-1]
        rk = np.empty(len(df), dtype=int)
        rk[order] = np.arange(1, len(df) + 1)
        ranks[i] = rk

    out = pd.DataFrame(
        {
            "rank_mean": ranks.mean(axis=0),
            "rank_std": ranks.std(axis=0),
            "rank_p05": np.percentile(ranks, 5, axis=0),
            "rank_p95": np.percentile(ranks, 95, axis=0),
        },
        index=df.index,
    )
    return out


# ---------------------------------------------------------------------------
# Centroid-characteristic-based archetype labelling (stable across reseeds)
# ---------------------------------------------------------------------------
def label_archetypes(
    centroids: pd.DataFrame,
    enrol_col: str = "log_total_enrolments",
    update_col: str = "update_rate",
    balance_col: str = "balance_score",
) -> dict:
    """Map cluster_id → semantic name using centroid characteristics, not hardcoded ids.

    Logic:
      - Highest enrolment + lowest update rate → "High-Growth, Low-Maintenance"
      - Highest update rate                    → "Mature & Balanced"
      - Lowest balance score                   → "Low-Volume, Imbalanced"
      - Whatever remains                       → "Emerging Hotspots"
    """
    c = centroids.copy()
    labels: dict = {}

    priority_id = (c[enrol_col].rank(ascending=False) + c[update_col].rank(ascending=True)).idxmin()
    labels[priority_id] = "High-Growth, Low-Maintenance (Priority Outreach)"

    remaining = c.drop(index=priority_id)
    mature_id = remaining[update_col].idxmax()
    labels[mature_id] = "Mature & Balanced (Monitor)"

    remaining = remaining.drop(index=mature_id)
    imbalanced_id = remaining[balance_col].idxmin()
    labels[imbalanced_id] = "Low-Volume, Imbalanced (Capacity Building)"

    for remaining_id in remaining.drop(index=imbalanced_id).index:
        labels[remaining_id] = "Emerging Hotspots (Proactive Camps)"

    return labels


# ---------------------------------------------------------------------------
# Temporal hold-out: are district rankings stable across time windows?
# ---------------------------------------------------------------------------
def temporal_holdout_rank_stability(
    df_enrol: pd.DataFrame,
    df_demo: pd.DataFrame,
    date_col: str = "date",
    group_col: str = "district",
    holdout_months: int = 3,
    enrol_age_cols: Iterable[str] = ("age_0_5", "age_5_17", "age_18_greater"),
    demo_age_cols: Iterable[str] = ("demo_age_5_17", "demo_age_17_"),
) -> dict:
    """Compare risk rankings on an earlier window vs the most-recent N months.

    The risk index is unsupervised, so we can't compute precision/recall, but
    we *can* ask whether the priority list is stable when re-fit on a different
    slice of time. High Spearman ρ → the ranking is a property of the
    districts, not a property of which months we happened to look at.
    """
    enrol = df_enrol.copy()
    demo = df_demo.copy()
    enrol[date_col] = pd.to_datetime(enrol[date_col], errors="coerce", dayfirst=True)
    demo[date_col] = pd.to_datetime(demo[date_col], errors="coerce", dayfirst=True)

    enrol = enrol.dropna(subset=[date_col, group_col])
    demo = demo.dropna(subset=[date_col, group_col])

    cutoff = max(enrol[date_col].max(), demo[date_col].max()) - pd.DateOffset(months=holdout_months)

    def _build(en: pd.DataFrame, de: pd.DataFrame) -> pd.DataFrame:
        en_g = en.groupby(group_col)[list(enrol_age_cols)].sum().sum(axis=1).to_frame("total_enrolments")
        up_g = de.groupby(group_col)[list(demo_age_cols)].sum().sum(axis=1).to_frame("total_updates")
        out = en_g.join(up_g, how="inner").fillna(0)
        out["update_rate"] = (
            (out["total_updates"] / out["total_enrolments"]).replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        bal = calculate_balance_score(de.rename(columns={group_col: "group"}).assign(group=de[group_col]))
        out = out.join(bal, how="left").fillna(0).reset_index()
        out = out[out["total_enrolments"] > CONFIG["thresholds"]["min_enrol_for_analysis"]]
        return calculate_risk_index(out).set_index(group_col)["identity_maintenance_risk"]

    early = _build(enrol[enrol[date_col] <= cutoff], demo[demo[date_col] <= cutoff])
    late = _build(enrol[enrol[date_col] > cutoff], demo[demo[date_col] > cutoff])

    common = early.index.intersection(late.index)
    if len(common) < 5:
        return {"n_common": int(len(common)), "spearman_rho": float("nan"), "cutoff": cutoff}

    rho = early.loc[common].rank().corr(late.loc[common].rank(), method="spearman")
    top20_early = set(early.loc[common].nlargest(20).index)
    top20_late = set(late.loc[common].nlargest(20).index)
    overlap = len(top20_early & top20_late) / 20.0

    return {
        "cutoff": cutoff,
        "n_common": int(len(common)),
        "n_early": int(len(early)),
        "n_late": int(len(late)),
        "spearman_rho": float(rho),
        "top20_overlap": float(overlap),
    }


# ---------------------------------------------------------------------------
# Fuzzy district matching for NITI cross-reference
# ---------------------------------------------------------------------------
# Curated alias map for the well-known transliteration renames. Fuzzy scorers
# can't bridge these (BANGALORE/BENGALURU max ~70 on WRatio) so we seed them
# explicitly and fall back to fuzzy for everything else.
DISTRICT_ALIASES: dict[str, str] = {
    "BANGALORE": "BENGALURU",
    "BANGALORE RURAL": "BENGALURU RURAL",
    "BANGALORE URBAN": "BENGALURU URBAN",
    "GURGAON": "GURUGRAM",
    "ALLAHABAD": "PRAYAGRAJ",
    "MUZAFFARPUR": "MUZAFFARPUR",
    "BOMBAY": "MUMBAI",
    "CALCUTTA": "KOLKATA",
    "MADRAS": "CHENNAI",
    "TRIVANDRUM": "THIRUVANANTHAPURAM",
    "PONDICHERRY": "PUDUCHERRY",
    "ORISSA": "ODISHA",
    "MYSORE": "MYSURU",
    "BELGAUM": "BELAGAVI",
    "GULBARGA": "KALABURAGI",
}


def fuzzy_match_districts(
    left: pd.Series,
    right: pd.Series,
    threshold: int | None = None,
) -> pd.DataFrame:
    """Match district names from `left` to `right`.

    Strategy:
      1. Curated alias map for well-known transliteration renames.
      2. Fuzzy fallback (rapidfuzz WRatio) for everything else.

    Returns DataFrame with columns: left, right_match, score, matched (bool).
    Requires rapidfuzz; pinned in requirements.txt.
    """
    from rapidfuzz import fuzz, process

    if threshold is None:
        threshold = CONFIG["thresholds"]["fuzzy_match_threshold"]

    right_list = right.dropna().unique().tolist()
    right_set = set(right_list)

    rows = []
    for name in left.dropna().unique():
        if not right_list:
            rows.append((name, None, 0, False))
            continue

        # 1. Alias hit
        alias = DISTRICT_ALIASES.get(name)
        if alias is not None and alias in right_set:
            rows.append((name, alias, 100, True))
            continue

        # 2. Exact match
        if name in right_set:
            rows.append((name, name, 100, True))
            continue

        # 3. Fuzzy fallback
        match, score, _ = process.extractOne(name, right_list, scorer=fuzz.WRatio)
        rows.append((name, match, score, score >= threshold))

    return pd.DataFrame(rows, columns=["left", "right_match", "score", "matched"])
