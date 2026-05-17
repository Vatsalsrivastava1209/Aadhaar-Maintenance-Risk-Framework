"""One-shot notebook surgery: apply the May-2026 fixes.

Run from repo root:
    python tools/apply_notebook_edits.py

Idempotent: re-running detects existing markers and skips already-applied edits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_PATH = Path("UIDAI Project.ipynb")

# Sentinels we put in cells we insert/modify so a re-run can detect prior edits.
SENT_IMPORT = "# >>> apply_notebook_edits: imports v1"
SENT_SCALER_MD = "# >>> apply_notebook_edits: scaler-justification v1"
SENT_BIAS = "# >>> apply_notebook_edits: bias-slice v1"
SENT_PERCAP = "# >>> apply_notebook_edits: per-capita v1"
SENT_TEMPORAL = "# >>> apply_notebook_edits: temporal-holdout v1"
SENT_NITI_CODE = "# >>> apply_notebook_edits: niti-stratified v1"
SENT_NITI_MD = "<!-- apply_notebook_edits: niti-md v1 -->"


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def cell_text(cell: dict) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else str(src)


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # ---- 1. Extend cell 1 imports ----------------------------------------
    cell1 = cell_text(cells[1])
    if SENT_IMPORT not in cell1:
        cell1 = cell1.replace(
            "    sensitivity_rank_stability,\n)",
            "    sensitivity_rank_stability,\n    risk_per_capita,\n    temporal_holdout_rank_stability,\n)\n"
            + SENT_IMPORT,
        )
        cells[1]["source"] = cell1.splitlines(keepends=True)
        print("[ok] extended cell 1 imports")
    else:
        print("[skip] cell 1 already has new imports")

    # ---- 2. Modify NITI code cell (find by content) ----------------------
    niti_code_idx = None
    for i, c in enumerate(cells):
        t = cell_text(c)
        if c.get("cell_type") == "code" and "NITI Aayog cross-reference" in t:
            niti_code_idx = i
            break
    if niti_code_idx is None:
        print("[err] could not locate NITI code cell")
        return 1

    niti_src = cell_text(cells[niti_code_idx])
    if SENT_NITI_CODE not in niti_src:
        # Append a within-state stratified comparison block before the scatter.
        insertion = (
            """

# ---------------------------------------------------------------------------
# Confounder control: within-state mean difference
# ---------------------------------------------------------------------------
# The unadjusted Welch's t compares aspirational vs standard districts across
# the whole country. But aspirational districts cluster in particular states,
# so the contrast partly reflects state-level effects (governance capacity,
# urbanisation, infrastructure) rather than aspirational status per se.
# Below we subtract each district's state mean before re-running the test,
# which sweeps out additive state effects.
"""
            + SENT_NITI_CODE
            + """
# state isn't on district_df (it's grouped by district only), so we look it up
# from df_master via a district -> mode(state) mapping built on the fly.
if "state" not in merged.columns:
    if "df_master" in dir() and "state" in df_master.columns and "district" in df_master.columns:
        dist_to_state = (
            df_master.dropna(subset=["district", "state"])
            .groupby("district")["state"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        )
        merged = merged.merge(
            dist_to_state.rename("state"), left_on="district", right_index=True, how="left",
        )

if "state" in merged.columns:
    merged["risk_demeaned"] = merged.groupby("state")["identity_maintenance_risk"].transform(
        lambda x: x - x.mean()
    )
    asp_dm = merged.loc[merged["is_aspirational"], "risk_demeaned"].dropna().to_numpy()
    std_dm = merged.loc[~merged["is_aspirational"], "risk_demeaned"].dropna().to_numpy()
    if len(asp_dm) > 1 and len(std_dm) > 1:
        t_dm, p_dm = stats.ttest_ind(asp_dm, std_dm, equal_var=False)
        diff_dm = asp_dm.mean() - std_dm.mean()
        print("\\nWithin-state stratified comparison (state fixed effect removed):")
        print(f"  Aspirational (n={len(asp_dm):,}): mean demeaned risk = {asp_dm.mean():+.4f}")
        print(f"  Standard     (n={len(std_dm):,}): mean demeaned risk = {std_dm.mean():+.4f}")
        print(f"  Difference:  Delta = {diff_dm:+.4f}   t = {t_dm:.3f}   p = {p_dm:.4f}")
        print("  Interpretation: a shrunken effect after demeaning means much of the")
        print("  raw aspirational-vs-standard gap was a state-composition effect.")
    else:
        print("\\n[stratified comparison skipped: not enough rows after dropna]")
else:
    print("\\n[stratified comparison skipped: state column not on merged frame]")
"""
        )
        # Insert before the scatter — find the 'fig = px.scatter' line.
        scatter_idx = niti_src.find("fig = px.scatter(")
        if scatter_idx == -1:
            niti_src = niti_src + insertion
        else:
            niti_src = niti_src[:scatter_idx] + insertion + "\n" + niti_src[scatter_idx:]
        cells[niti_code_idx]["source"] = niti_src.splitlines(keepends=True)
        print(f"[ok] inserted stratified comparison into cell {niti_code_idx}")
    else:
        print("[skip] NITI code already has stratified comparison")

    # ---- 3. Reframe NITI markdown cell -----------------------------------
    niti_md_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "markdown" and "Higher Risk in Aspirational Districts" in cell_text(c):
            niti_md_idx = i
            break
    if niti_md_idx is None:
        print("[warn] could not locate NITI markdown cell")
    else:
        niti_md_src = cell_text(cells[niti_md_idx])
        if SENT_NITI_MD not in niti_md_src:
            new_md = (
                SENT_NITI_MD
                + """
### Aspirational districts and maintenance risk: a development-gap pattern, not a causal claim

The cell above reports two comparisons:

1. **Unadjusted Welch's t-test** with 95% CI on the mean difference and Hedges' *g* effect size.
2. **Within-state stratified comparison** that subtracts each district's state mean before re-testing.

We report both deliberately. Aspirational districts are aspirational *because* they're structurally underdeveloped — they concentrate in particular states with lower governance capacity, lower urbanisation, and smaller administrative budgets. A raw aspirational-vs-standard gap therefore conflates the aspirational designation with the state-level context. The stratified Delta is the part that survives once you sweep out additive state effects, and it is the honest number to read.

**Framing**: the finding is *consistent with the broader development gap that the Aspirational Districts programme was designed to address* — not evidence that the designation itself causes the higher risk. A causal claim would require a matched comparison (propensity-score matching on state, urbanisation, population) or a quasi-experimental design.

### Policy implications (read with the caveat above)
- **Bundle interventions** with existing Aspirational programmes (Financial Inclusion, Skill Development) where field infrastructure already exists.
- **Prioritise** the "High-Growth, Low-Maintenance" and "Low-Volume, Imbalanced" archetypes inside the 112 Aspirational list.
- **Caveat on coverage**: fuzzy district-name matching excludes unmatched names from the comparison and may bias the test.
"""
            )
            cells[niti_md_idx]["source"] = new_md.splitlines(keepends=True)
            print(f"[ok] reframed NITI markdown cell {niti_md_idx}")
        else:
            print("[skip] NITI markdown already reframed")

    # ---- 4. Insert scaler-justification markdown before the K-search cell ----
    ksearch_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and "K-search" in cell_text(c) and "silhouette" in cell_text(c):
            ksearch_idx = i
            break
    already_has_scaler_md = any(SENT_SCALER_MD in cell_text(c) for c in cells)
    if ksearch_idx is not None and not already_has_scaler_md:
        scaler_md = (
            SENT_SCALER_MD
            + """
### A note on feature scaling for KMeans

KMeans uses Euclidean distance, so unscaled features with larger numeric ranges dominate the distance metric. Our five clustering features have wildly different scales (e.g. `log_total_enrolments` is in [~5, ~17] while `update_rate` is in [0, 1]), so scaling is not optional.

We use **MinMax** rather than **Standard** scaling for two reasons:

1. **Consistency with the risk index.** `calculate_risk_index` MinMax-scales its inputs, so the clustering view sits in the same `[0, 1]` space as the score the user will see on the map.
2. **Distribution shape.** `update_rate` and `balance_score` are bounded by construction; MinMax preserves the bounded interpretation. StandardScaler would push them into z-space and assume Gaussianity that isn't really there.

The bootstrap-ARI check below confirms cluster assignments are stable under resampling. A reviewer who prefers StandardScaler can swap one line in the next cell and re-run; the index pipeline is unaffected.
"""
        )
        cells.insert(ksearch_idx, md_cell(scaler_md))
        print(f"[ok] inserted scaler-justification markdown before cell {ksearch_idx}")
    else:
        print("[skip] scaler-justification already present or K-search not found")

    # ---- 5. Add three new cells after sanity-check cell 14 ---------------
    sanity_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and "Sanity check 1: does the composite index beat" in cell_text(c):
            sanity_idx = i
            break
    if sanity_idx is None:
        print("[err] could not locate sanity-check cell")
        return 1

    have_bias = any(SENT_BIAS in cell_text(c) for c in cells)
    have_percap = any(SENT_PERCAP in cell_text(c) for c in cells)
    have_temporal = any(SENT_TEMPORAL in cell_text(c) for c in cells)

    new_cells = []

    if not have_bias:
        new_cells.extend(
            [
                md_cell("""## Bias slice: how is risk distributed across regions?

A risk score that systematically over-flags structurally disadvantaged regions can do harm even with good intentions. We slice the risk distribution two ways:

1. **By state** — boxplot of `identity_maintenance_risk` per state. A flat distribution would mean state membership doesn't predict risk; concentration in particular states is a flag.
2. **By enrolment-volume quartile** — used as a crude rural/urban proxy in the absence of a Census-linked classification. A monotone trend would mean the model is mostly tracking population size, which we already know is partially baked into the Impact term.

The figure is saved to `plots/bias_slice.png` and referenced from the README's *Ethical considerations* section."""),
                code_cell(
                    SENT_BIAS
                    + """
import matplotlib.pyplot as plt

if "state" in district_df.columns:
    state_col = "state"
elif "STATE" in district_df.columns:
    state_col = "STATE"
else:
    state_col = None

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

if state_col is not None:
    top_states = (
        district_df.groupby(state_col)["identity_maintenance_risk"].median().sort_values(ascending=False).head(15).index
    )
    sub = district_df[district_df[state_col].isin(top_states)]
    order = sub.groupby(state_col)["identity_maintenance_risk"].median().sort_values(ascending=False).index
    sns.boxplot(
        data=sub, x=state_col, y="identity_maintenance_risk", order=order, ax=axes[0], color="steelblue",
    )
    axes[0].set_title("Risk distribution by state (top 15 by median)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("identity_maintenance_risk")
    axes[0].tick_params(axis="x", rotation=75)
else:
    axes[0].text(0.5, 0.5, "no state column on district_df", ha="center", va="center")
    axes[0].set_axis_off()

# Enrolment-volume quartile as rural/urban proxy
district_df["enrol_quartile"] = pd.qcut(
    district_df["total_enrolments"], q=4,
    labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"],
)
sns.boxplot(
    data=district_df, x="enrol_quartile", y="identity_maintenance_risk", ax=axes[1], color="indianred",
)
axes[1].set_title("Risk by enrolment-volume quartile (proxy for rural/urban)")
axes[1].set_xlabel("Enrolment quartile")

plt.tight_layout()
import os
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/bias_slice.png", dpi=120, bbox_inches="tight")
plt.show()

print("\\nMedian risk by enrolment quartile (a strong monotone trend would mean")
print("the index is mostly recovering population size — read alongside per-capita view below):")
print(district_df.groupby("enrol_quartile")["identity_maintenance_risk"].median().round(4))
"""
                ),
            ]
        )
        print("[ok] queued bias-slice cells")

    if not have_percap:
        new_cells.extend(
            [
                md_cell("""## Per-capita view: are we finding risk, or finding population?

The composite index multiplies P(failure) by Impact (`log1p(total_enrolments)`), so high-volume districts have a mechanical advantage. A reviewer should ask: if we strip the volume term, does the priority list change?

We don't have a Census-linked population per district available in this repo, so we use `total_enrolments` itself as a population proxy. With that proxy, dividing the composite by `log1p(total_enrolments)` collapses to the `p_failure` term — same ranking, clearer semantics. `risk_per_capita` returns it directly.

Use this as a **secondary** ranking: it's the "where would you go if you ignored how many people lived there?" view. The composite remains the primary triage signal because impact matters for resource allocation."""),
                code_cell(
                    SENT_PERCAP
                    + """
district_df["risk_per_capita"] = risk_per_capita(district_df).values

top20_composite = district_df.nlargest(20, "identity_maintenance_risk")[
    ["district", "total_enrolments", "update_rate", "identity_maintenance_risk"]
].reset_index(drop=True)
top20_percap = district_df.nlargest(20, "risk_per_capita")[
    ["district", "total_enrolments", "update_rate", "risk_per_capita"]
].reset_index(drop=True)

side_by_side = pd.concat(
    [top20_composite, top20_percap],
    axis=1,
    keys=["By composite (volume-weighted)", "By per-capita (volume-stripped)"],
)
print("Top-20 districts under two rankings:")
print(side_by_side.to_string())

overlap = len(set(top20_composite["district"]) & set(top20_percap["district"]))
print(f"\\nOverlap between the two top-20 lists: {overlap} / 20")
print("(Low overlap means the volume term is doing real work — and that the per-capita view")
print(" surfaces a different set of districts a triage tool should be aware of.)")
"""
                ),
            ]
        )
        print("[ok] queued per-capita cells")

    if not have_temporal:
        new_cells.extend(
            [
                md_cell("""## Temporal hold-out: are the rankings stable across time windows?

The risk index is unsupervised so we can't compute precision/recall. But we can ask: if we re-fit on a different slice of time, does the priority list look the same? A high Spearman correlation between an earlier-window ranking and a recent-window ranking means the index is picking up a property of the districts, not of which months we happened to look at."""),
                code_cell(
                    SENT_TEMPORAL
                    + """
holdout = temporal_holdout_rank_stability(
    df_enrol=df_master,
    df_demo=df_master2,
    date_col="date",
    group_col="district",
    holdout_months=3,
)
print("Temporal hold-out (most-recent 3 months held out):")
for k, v in holdout.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")
print("\\nRho close to 1.0  ->  ranking is stable across the time split.")
print("Top-20 overlap close to 1.0  ->  the priority list itself transfers between windows.")
"""
                ),
            ]
        )
        print("[ok] queued temporal-hold-out cells")

    if new_cells:
        insert_at = sanity_idx + 1
        cells[insert_at:insert_at] = new_cells
        print(f"[ok] inserted {len(new_cells)} new cells at index {insert_at}")
    else:
        print("[skip] bias/percap/temporal cells already present")

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {NB_PATH} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
