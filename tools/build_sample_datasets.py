"""Build a tiny anonymised sample under datasets/sample/.

Goal: a fresh-clone of the repo can run `streamlit run app.py` and see a
real-looking risk map without needing the full UIDAI extracts.

Anonymisation:
- Keep state names (public information; needed for the choropleth join).
- Keep district names that match `india_districts.json` keys (needed for the
  choropleth join — otherwise the map renders blank).
- Replace pincodes with stable surrogate codes (`SAMPLE_PIN_0001`, ...).
- Sample ~3,000 rows per source file, stratified by state so every state has
  at least one row where possible.

Run from repo root:
    python tools/build_sample_datasets.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

RAW_BASE = Path("datasets")
OUT_BASE = Path("datasets/sample")
# Target ~80 districts with enough volume to clear min_enrol_for_analysis (1000)
# in the demo. ~25k rows per file lands the largest files under ~1.5MB,
# comfortably below the 2MB pre-commit guard.
ROWS_PER_FILE = 25000
TOP_DISTRICTS = 80
GEOJSON_PATH = Path("india_districts.json")
RANDOM_STATE = 42


def load_concat(folder: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(folder / "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSVs under {folder}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def stratified_sample(df: pd.DataFrame, n: int, strat_col: str, seed: int = RANDOM_STATE) -> pd.DataFrame:
    if strat_col not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=seed)
    groups = df.groupby(strat_col, group_keys=False)
    per_group = max(1, n // max(1, df[strat_col].nunique()))
    sampled = groups.apply(lambda g: g.sample(n=min(per_group, len(g)), random_state=seed))
    if len(sampled) < n:
        extra = df.drop(index=sampled.index).sample(
            n=min(n - len(sampled), len(df) - len(sampled)), random_state=seed
        )
        sampled = pd.concat([sampled, extra])
    return sampled.head(n).reset_index(drop=True)


def main() -> int:
    if not RAW_BASE.exists():
        print("[err] datasets/ not present — run from a clone that has the real extracts")
        return 1

    # Load district whitelist from the geojson — only districts present in the
    # choropleth keys will render, so we restrict the sample to those.
    if GEOJSON_PATH.exists():
        with GEOJSON_PATH.open(encoding="utf-8") as f:
            geo = json.load(f)
        valid_districts = {
            feat["properties"].get("district", "").upper().strip() for feat in geo.get("features", [])
        }
        valid_districts.discard("")
        print(f"[info] {len(valid_districts):,} districts in geojson")
    else:
        valid_districts = set()
        print("[warn] no geojson — sample will not be choropleth-filtered")

    folders = [
        "api_data_aadhar_enrolment",
        "api_data_aadhar_biometric",
        "api_data_aadhar_demographic",
    ]

    # Pincode surrogate map shared across the three folders so the
    # pincode→district join in app.py still works inside the sample.
    pincode_map: dict[str, str] = {}

    # First pass: pick the top-K districts by row count in the enrolment file
    # so we get realistic per-district volume in the demo. We use the same
    # district whitelist across all three folders.
    top_districts: set[str] | None = None
    try:
        enr_raw = load_concat(RAW_BASE / folders[0])
        if valid_districts and "district" in enr_raw.columns:
            up = enr_raw["district"].astype(str).str.upper().str.strip()
            enr_raw = enr_raw[up.isin(valid_districts)]
        top_districts = set(enr_raw["district"].value_counts().head(TOP_DISTRICTS).index.tolist())
        print(f"[info] selected top-{TOP_DISTRICTS} districts by row count")
    except FileNotFoundError:
        pass

    for folder in folders:
        src = RAW_BASE / folder
        dst = OUT_BASE / folder
        dst.mkdir(parents=True, exist_ok=True)

        try:
            df = load_concat(src)
        except FileNotFoundError as e:
            print(f"[skip] {folder}: {e}")
            continue

        # Restrict to choropleth-valid districts when possible
        if valid_districts and "district" in df.columns:
            up = df["district"].astype(str).str.upper().str.strip()
            df = df[up.isin(valid_districts)].copy()

        # Further restrict to the top-K districts so per-district volume is
        # realistic enough to clear the demo's min_enrol_for_analysis threshold.
        if top_districts and "district" in df.columns:
            df = df[df["district"].isin(top_districts)].copy()

        sample = stratified_sample(df, ROWS_PER_FILE, strat_col="district")

        # Anonymise pincodes deterministically
        if "pincode" in sample.columns:
            sample["pincode"] = sample["pincode"].astype(str)
            for pin in sample["pincode"].unique():
                if pin not in pincode_map:
                    pincode_map[pin] = f"SAMPLE_PIN_{len(pincode_map) + 1:05d}"
            sample["pincode"] = sample["pincode"].map(pincode_map)

        out_file = dst / f"{folder}_sample.csv"
        sample.to_csv(out_file, index=False)
        print(f"[ok] {folder}: {len(sample):,} rows -> {out_file}")

    # Sample pincode_directory.csv keyed to the surrogate pincodes we just emitted.
    pin_src = Path("pincode_directory.csv")
    if pin_src.exists() and pincode_map:
        pin_df = pd.read_csv(pin_src, dtype={"pincode": str})
        # Build a sample-pincode → district map by mapping real pincodes that
        # *were* in our sampled rows to the original district/state in the
        # original directory.
        pin_lookup = pin_df.drop_duplicates(subset="pincode").set_index("pincode")
        records = []
        for real_pin, surrogate in pincode_map.items():
            if real_pin in pin_lookup.index:
                row = pin_lookup.loc[real_pin]
                records.append(
                    {
                        "pincode": surrogate,
                        "district": row.get("district", ""),
                        "state": row.get("state", ""),
                    }
                )
        if records:
            sample_pin = pd.DataFrame(records).drop_duplicates(subset="pincode")
            sample_pin.to_csv(OUT_BASE / "pincode_directory_sample.csv", index=False)
            print(
                f"[ok] pincode_directory: {len(sample_pin):,} rows -> {OUT_BASE / 'pincode_directory_sample.csv'}"
            )

    # Save the pincode mapping audit trail (handy for verifying anonymisation)
    if pincode_map:
        audit = pd.DataFrame(sorted(pincode_map.items()), columns=["real_pincode", "surrogate"])
        audit_path = OUT_BASE / "_anonymisation_audit.csv"
        audit.to_csv(audit_path, index=False)
        # The audit file should NOT be committed; it maps surrogate → real pin.
        # We add it to .gitignore in the same change-set.
        print(f"[ok] audit: {len(audit):,} rows -> {audit_path} (gitignored)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
