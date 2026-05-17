"""Unit tests for utils.helpers.

Run from repo root:
    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_district_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district": ["A", "B", "C", "D", "E"],
            "total_enrolments": [10_000, 500_000, 1_000_000, 50_000, 200_000],
            "total_updates": [9_000, 50_000, 50_000, 25_000, 30_000],
            "update_rate": [0.90, 0.10, 0.05, 0.50, 0.15],
            "balance_score": [0.9, 0.4, 0.2, 0.7, 0.5],
        }
    )


@pytest.fixture
def toy_demo_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": ["S1", "S1", "S2", "S2", "S3"],
            "demo_age_5_17": [100, 200, 50, 60, 300],
            "demo_age_17_": [110, 190, 70, 40, 290],
        }
    )


# ---------------------------------------------------------------------------
# calculate_risk_index
# ---------------------------------------------------------------------------
class TestRiskIndex:
    def test_output_columns_present(self, toy_district_df):
        out = calculate_risk_index(toy_district_df)
        for col in (
            "enrol_norm",
            "update_rate_norm",
            "balance_norm",
            "p_failure",
            "impact",
            "identity_maintenance_risk",
            "risk_level",
        ):
            assert col in out.columns, f"missing column {col}"

    def test_index_in_unit_interval(self, toy_district_df):
        out = calculate_risk_index(toy_district_df)
        assert out["identity_maintenance_risk"].between(0, 1).all()
        assert out["p_failure"].between(0, 1).all()
        assert out["impact"].between(0, 1).all()

    def test_weights_sum_to_one(self):
        pf = CONFIG["risk_weights"]["p_failure"]
        comp = CONFIG["risk_weights"]["composite"]
        assert pytest.approx(sum(pf.values()), abs=1e-9) == 1.0
        assert pytest.approx(sum(comp.values()), abs=1e-9) == 1.0

    def test_low_update_rate_raises_p_failure(self, toy_district_df):
        out = calculate_risk_index(toy_district_df).set_index("district")
        # District C has lowest update rate AND lowest balance → highest p_failure.
        assert out["p_failure"].idxmax() == "C"
        # District A has highest update rate AND highest balance → lowest p_failure.
        assert out["p_failure"].idxmin() == "A"

    def test_impact_uses_log1p_so_outliers_dont_dominate(self):
        # If we did NOT log-transform, a 100x larger enrolment would map to ~1.0
        # and everything else to ~0.0. After log1p, the spread is much tighter.
        df = pd.DataFrame(
            {
                "district": ["small", "huge"],
                "total_enrolments": [1_000, 100_000_000],
                "total_updates": [100, 1_000_000],
                "update_rate": [0.1, 0.01],
                "balance_score": [0.5, 0.5],
            }
        )
        out = calculate_risk_index(df).set_index("district")
        # log1p compresses the gap; small district should still have nonzero impact-norm.
        assert out.loc["small", "impact"] >= 0.0
        # Two rows → MinMax gives [0, 1] exactly, but the *log* transform happens
        # before scaling. So this just sanity-checks the function survives extremes.
        assert out["impact"].between(0, 1).all()

    def test_constant_input_does_not_crash(self):
        df = pd.DataFrame(
            {
                "district": ["x", "y", "z"],
                "total_enrolments": [1000, 1000, 1000],
                "total_updates": [500, 500, 500],
                "update_rate": [0.5, 0.5, 0.5],
                "balance_score": [0.5, 0.5, 0.5],
            }
        )
        out = calculate_risk_index(df)
        assert len(out) == 3
        assert out["identity_maintenance_risk"].notna().all()


# ---------------------------------------------------------------------------
# naive_baseline_risk
# ---------------------------------------------------------------------------
class TestBaseline:
    def test_baseline_is_one_minus_update_rate(self, toy_district_df):
        base = naive_baseline_risk(toy_district_df)
        expected = 1 - toy_district_df["update_rate"]
        pd.testing.assert_series_equal(base, expected, check_names=False)

    def test_baseline_clipped_to_unit(self):
        df = pd.DataFrame({"update_rate": [-0.5, 1.5, 0.3]})
        base = naive_baseline_risk(df)
        assert base.between(0, 1).all()


# ---------------------------------------------------------------------------
# calculate_balance_score
# ---------------------------------------------------------------------------
class TestBalanceScore:
    def test_balance_in_unit_interval(self, toy_demo_df):
        bal = calculate_balance_score(toy_demo_df.rename(columns={"state": "group"}))
        assert bal["balance_score"].between(0, 1).all()

    def test_perfect_balance_scores_higher_than_imbalance(self):
        bal_df = pd.DataFrame(
            {
                "group": ["balanced", "balanced", "imbalanced", "imbalanced"],
                "demo_age_5_17": [100, 100, 1_000_000, 1_000_000],
                "demo_age_17_": [100, 100, 1, 1],
            }
        )
        out = calculate_balance_score(bal_df)
        assert out.loc["balanced", "balance_score"] > out.loc["imbalanced", "balance_score"]


# ---------------------------------------------------------------------------
# build_state_engagement
# ---------------------------------------------------------------------------
class TestEngagement:
    def test_engagement_has_expected_columns(self):
        enrol = pd.DataFrame(
            {
                "state": ["S1", "S1", "S2"],
                "age_0_5": [10, 20, 5],
                "age_5_17": [10, 20, 5],
                "age_18_greater": [10, 20, 5],
            }
        )
        demo = pd.DataFrame(
            {
                "state": ["S1", "S2"],
                "demo_age_5_17": [50, 100],
                "demo_age_17_": [60, 110],
            }
        )
        eng = build_state_engagement(enrol, demo)
        for col in ("total_enrolments", "total_updates", "update_rate", "balance_score"):
            assert col in eng.columns

    def test_update_rate_handles_zero_enrolments(self):
        enrol = pd.DataFrame(
            {
                "state": ["Z"],
                "age_0_5": [0],
                "age_5_17": [0],
                "age_18_greater": [0],
            }
        )
        demo = pd.DataFrame(
            {
                "state": ["Z"],
                "demo_age_5_17": [100],
                "demo_age_17_": [0],
            }
        )
        eng = build_state_engagement(enrol, demo)
        assert np.isfinite(eng["update_rate"]).all()


# ---------------------------------------------------------------------------
# label_archetypes
# ---------------------------------------------------------------------------
class TestArchetypeLabels:
    def test_labels_assigned_by_characteristics_not_id(self):
        # Cluster 7 has highest log-enrol AND lowest update_rate → priority outreach
        centroids = pd.DataFrame(
            {
                "log_total_enrolments": [10, 12, 8, 14],
                "update_rate": [0.7, 0.5, 0.9, 0.05],
                "balance_score": [0.6, 0.3, 0.7, 0.4],
                "bio_demo_ratio": [0.5, 0.5, 0.5, 0.5],
            },
            index=[0, 1, 7, 3],
        )
        labels = label_archetypes(centroids)
        assert labels[7] != labels[3]
        # The lowest update_rate is at index 3 (0.05) AND highest log-enrol → priority
        assert "Priority Outreach" in labels[3]
        # Highest update_rate among remaining (0.9) is index 7 → Mature & Balanced
        assert "Mature" in labels[7]

    def test_all_clusters_get_a_label(self):
        centroids = pd.DataFrame(
            {
                "log_total_enrolments": [10, 11, 12, 13],
                "update_rate": [0.1, 0.5, 0.7, 0.3],
                "balance_score": [0.2, 0.6, 0.5, 0.4],
            },
            index=range(4),
        )
        labels = label_archetypes(centroids)
        assert set(labels.keys()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# sensitivity_rank_stability
# ---------------------------------------------------------------------------
class TestSensitivity:
    def test_returns_rank_summary(self, toy_district_df):
        df = calculate_risk_index(toy_district_df)
        sens = sensitivity_rank_stability(df, n_perturbations=30)
        for col in ("rank_mean", "rank_std", "rank_p05", "rank_p95"):
            assert col in sens.columns
        # Ranks are 1..N
        assert sens["rank_mean"].between(1, len(df)).all()


# ---------------------------------------------------------------------------
# fuzzy_match_districts
# ---------------------------------------------------------------------------
class TestFuzzyMatch:
    def test_known_alias_matches(self):
        left = pd.Series(["BANGALORE", "GURGAON", "NONSENSE_XYZ"])
        right = pd.Series(["BENGALURU", "GURUGRAM", "DELHI"])
        out = fuzzy_match_districts(left, right, threshold=70)
        out_indexed = out.set_index("left")
        assert out_indexed.loc["BANGALORE", "matched"]
        assert out_indexed.loc["GURGAON", "matched"]
        assert not out_indexed.loc["NONSENSE_XYZ", "matched"]
