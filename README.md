# Aadhaar Identity Maintenance Risk Framework

> District-level prioritisation engine that uses public Aadhaar
> enrolment / update data to flag where stale records are most likely to
> cause authentication failures and exclusion.

[![CI](https://github.com/Vatsalsrivastava1209/Uidai-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Vatsalsrivastava1209/Uidai-project/actions/workflows/ci.yml)

---

## The problem

UIDAI publishes monthly enrolment and update data, but operationally the
question for any state IT department is **"where do we send the next mobile
update camp?"** Districts with high enrolment volume and a low update rate
accumulate stale biometric / demographic records, which cause downstream
authentication failures — and the people most affected are the ones who use
Aadhaar most often for entitlements.

The dataset answers "what's the volume" but not "where's the risk".

## What this project does

1. **Aggregates** enrolment, biometric, and demographic update records to the
   district level via a pincode → district mapping.
2. **Builds a composite risk index** with an explicit *Risk = P(failure) ×
   Impact* decomposition so each component can be argued with separately.
3. **Clusters** districts into archetypes on an enriched feature set
   (log-enrolment, update rate, balance, bio/demo ratio, growth slope) that
   does **not** reuse the index inputs — so the clusters describe something
   the index doesn't already encode.
4. **Forecasts** monthly update load per archetype with Prophet, and reports
   the **rolling-horizon MAPE** so the forecast comes with an error bar
   instead of a vibe.
5. **Cross-validates against NITI Aayog** Aspirational Districts data using a
   curated alias map + fuzzy district matching, with a Welch's t-test +
   Hedges' *g* + 95% CI on the mean difference (not just two means quoted in
   bold).

## Headline findings

- **Risk concentrates outside the obvious places.** The composite index
  reranks districts vs the naive `1 - update_rate` baseline; Spearman ρ and
  top-20 overlap are reported in the notebook.
- **Aspirational districts carry measurably higher maintenance risk.**
  The notebook reports the t-statistic, p-value, effect size, and CI —
  read the numbers before believing them.
- **Update demand is rising, but the validated forecast horizon is short.**
  Per-archetype MAPE is printed alongside every projection.

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Aggregation | pandas | Standard |
| Modelling   | scikit-learn (KMeans + sensitivity), Prophet | Interpretable, hackable, plays well with policy audiences |
| Validation  | scipy.stats, rapidfuzz | Welch's t-test + alias-aware fuzzy matching |
| Maps        | Plotly choropleth | One file, GitHub-renderable |
| Demo        | Streamlit | One-command shareable app |
| Hygiene     | ruff, pytest, nbstripout, pre-commit | Keeps the notebook reviewable |

## Run it

```bash
# 1. Clone
git clone https://github.com/Vatsalsrivastava1209/Uidai-project.git
cd Uidai-project

# 2. Install (pinned versions)
pip install -r requirements.txt

# 3a. The notebook
jupyter lab "UIDAI Project.ipynb"

# 3b. The Streamlit demo
streamlit run app.py

# 4. Tests + lint (dev workflow)
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

**Note on data**: the raw CSV folders under `datasets/` are gitignored
because of size. The notebook documents the expected schema in cell 2;
the same UIDAI public extracts you can pull from the official portal will
plug in unchanged.

## Repository layout

```
.
├── UIDAI Project.ipynb       # Main analysis (stripped of outputs, ~56KB)
├── app.py                    # Streamlit demo (uses utils/)
├── utils/
│   ├── config.py             # Tunable weights + thresholds
│   └── helpers.py            # All numerical logic + sensitivity analysis
├── tests/test_helpers.py     # Pytest cases for utils/
├── tools/rewrite_notebook.py # Notebook surgery script (one-time)
├── .github/workflows/ci.yml  # Lint + tests + notebook-size guard
├── .pre-commit-config.yaml   # ruff + nbstripout
├── ARCHITECTURE.md           # Productionisation story
├── financial_inclusion.csv   # NITI Aayog reference
├── pincode_directory.csv     # Pincode → district map
├── india_districts.json      # GeoJSON for choropleth
└── plots/                    # Exported figures (PNG)
```

## Limitations

Read these before quoting the findings.

- **No ground-truth labels.** The risk index is built from operational
  proxies (update rate, balance, volume), not from actual authentication
  failure incidence. Without a labelled set of "districts where Aadhaar
  failures occurred at rate X", the index is plausible but not validated.
- **Weights are heuristic.** Composite weights in `utils/config.py` are
  defensible defaults, not learned from data. The sensitivity analysis in
  the notebook shows the *robustness* of rankings under ±15% jitter, but
  cannot tell you the weights are *right*.
- **District-name joins are imperfect.** Even with a curated alias map
  plus rapidfuzz fallback, some NITI / Aadhaar district names do not
  match cleanly. Coverage is reported in the notebook; a proper fix is to
  migrate everything to LGD district codes.
- **Forecast horizon is short.** Prophet CV is honest about MAPE per
  archetype. Do not extrapolate beyond the validated horizon.
- **Population denominator is missing.** Risk per capita would be more
  honest than absolute volume — the index treats "high enrolment" as
  impact rather than normalising by population. This is intentional for a
  triage tool but should be acknowledged.

## Ethical considerations

This project risk-scores **regions, not individuals**, and uses only
aggregated public data. Even so:

- **Stigmatisation risk**: any geographic risk model can be misused to
  justify resource withdrawal from "risky" areas. The intended use is the
  opposite — prioritise *more* outreach to high-risk districts.
- **Disparate impact**: risk distributions should be audited by state,
  urban / rural classification, and Scheduled-Area status to ensure the
  model is not systematically over-flagging structurally disadvantaged
  regions purely because of population density. `ARCHITECTURE.md`
  describes the quarterly audit task.
- **Privacy**: no PII is used or required. Inputs are public aggregates.

## What I'd do with another two weeks

1. **Acquire authentication-failure labels** from UIDAI ops and learn the
   weights (logistic regression / gradient-boosted classifier) rather
   than heuristically setting them.
2. **Causal angle**: where mobile camps have been deployed historically,
   estimate the lift on update rate using propensity-score matching or a
   regression-discontinuity around the risk threshold.
3. **Per-capita normalisation** using 2011 Census + extrapolated district
   population, so "impact" is shifted from raw enrolment count to share
   of population needing service.
4. **Replace district-name joins with LGD codes** end-to-end.
5. **Ship the FastAPI scoring service** sketched in `ARCHITECTURE.md`.

## Acknowledgements

UIDAI for the open enrolment/update data, NITI Aayog for the Champions of
Change dashboard data, and India Post for the pincode directory.

Issues / feedback welcome via GitHub.
