"""Central place for all tunable parameters."""

from __future__ import annotations

CONFIG: dict = {
    # Risk = P(failure) × Impact framing.
    # P(failure) is built from update_rate (low rate → high failure prob) and
    # demographic balance (imbalanced age coverage → higher exclusion risk).
    # Impact is enrolment volume (more residents → more citizens affected).
    "risk_weights": {
        "p_failure": {  # weights inside the P(failure) sub-score (sum to 1.0)
            "update_rate": 0.55,
            "balance": 0.45,
        },
        "composite": {  # weights combining sub-scores into the composite (sum to 1.0)
            "p_failure": 0.65,
            "impact": 0.35,
        },
    },
    "clustering": {
        "n_clusters_search": list(range(2, 11)),  # search K∈{2..10}, not {2..5}
        "n_clusters_default": 4,
        # NOTE: features for clustering deliberately include richer signal beyond
        # the index inputs to avoid tautological "clusters recover the index".
        "features": [
            "log_total_enrolments",
            "update_rate",
            "balance_score",
            "bio_demo_ratio",  # mix of biometric vs demographic updates
            "enrol_growth_slope",  # 6-month enrolment trend
        ],
        "n_bootstrap_stability": 30,
        "random_state": 42,
    },
    "thresholds": {
        "min_enrol_for_analysis": 1000,
        "high_enrol_quantile": 0.7,
        "low_update_quantile": 0.3,
        "fuzzy_match_threshold": 88,  # rapidfuzz score ≥ this counts as a match
    },
    "forecasting": {
        "horizon_months": 12,
        "cv_initial_months": 18,
        "cv_period_months": 3,
        "cv_horizon_months": 6,
    },
    "files": {
        "pincode_mapping": "pincode_directory.csv",
        "district_geojson": "india_districts.json",
        "financial_inclusion": "financial_inclusion.csv",
    },
}
