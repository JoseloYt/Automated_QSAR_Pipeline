"""
Step 1 — Fetch bioactivity data from ChEMBL for a given receptor.

Usage:
    python step1_fetch_chembl.py --receptor FGFR1
"""

import argparse
import os
import sys

import pandas as pd
from chembl_webresource_client.new_client import new_client

import config


def search_target(receptor_name: str) -> pd.DataFrame:
    """Search ChEMBL for targets matching *receptor_name*."""
    target_api = new_client.target
    query = target_api.search(receptor_name)
    targets = pd.DataFrame.from_dict(query)
    if targets.empty:
        raise ValueError(f"No targets found for '{receptor_name}'")
    return targets


def select_best_target(targets: pd.DataFrame) -> str:
    """
    Auto-select the best Homo sapiens SINGLE PROTEIN target.
    Returns the target_chembl_id.
    """
    mask = (
        (targets["organism"] == config.ORGANISM)
        & (targets["target_type"] == config.TARGET_TYPE)
    )
    filtered = targets[mask].copy()

    if filtered.empty:
        # Fallback: just pick Homo sapiens regardless of target_type
        filtered = targets[targets["organism"] == config.ORGANISM].copy()
        if filtered.empty:
            raise ValueError(
                f"No {config.ORGANISM} targets found. "
                f"Available organisms: {targets['organism'].unique().tolist()}"
            )

    # Pick highest score
    filtered["score"] = pd.to_numeric(filtered["score"], errors="coerce")
    best = filtered.sort_values("score", ascending=False).iloc[0]
    chembl_id = best["target_chembl_id"]
    pref_name = best["pref_name"]
    print(f"  Selected target: {pref_name} ({chembl_id}), score={best['score']}")
    return chembl_id


def fetch_activities(
    target_chembl_id: str, bioactivity_type: str
) -> pd.DataFrame:
    """Fetch activity data from ChEMBL for a target + bioactivity type."""
    activity_api = new_client.activity
    results = activity_api.filter(
        target_chembl_id=target_chembl_id,
        standard_type=bioactivity_type,
    )
    df = pd.DataFrame.from_dict(results)
    if df.empty:
        raise ValueError(
            f"No activity data found for {target_chembl_id} "
            f"with standard_type='{bioactivity_type}'"
        )
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning of the raw ChEMBL data."""
    # Keep only rows with valid standard_value and canonical_smiles
    df = df[df["standard_value"].notna()].copy()
    df = df[df["canonical_smiles"].notna()].copy()

    # Convert standard_value to numeric
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"].notna()].copy()

    # Keep only rows with '=' relation (exact measurements)
    if "standard_relation" in df.columns:
        has_relation = df["standard_relation"].notna()
        exact = df["standard_relation"] == "="
        df = df[~has_relation | exact].copy()

    print(f"  Rows after cleaning: {len(df)}")
    return df


def run(receptor_name: str, bioactivity_type: str | None = None) -> str:
    """
    Main entry point for Step 1.
    Returns the path to the saved raw CSV.
    """
    bio_type = bioactivity_type or config.BIOACTIVITY_TYPE
    out_dir = config.get_output_dir(receptor_name)

    print(f"[Step 1] Fetching ChEMBL data for '{receptor_name}' "
          f"(bioactivity_type={bio_type})")

    # 1. Search targets
    print("  Searching targets...")
    targets = search_target(receptor_name)
    print(f"  Found {len(targets)} targets")

    # 2. Select best target
    target_id = select_best_target(targets)

    # 3. Fetch activities
    print(f"  Fetching activities (type={bio_type})...")
    df = fetch_activities(target_id, bio_type)
    print(f"  Retrieved {len(df)} activity records")

    # 4. Basic clean
    df = clean_raw_data(df)

    # 5. Select key columns and save
    keep_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_type",
        "standard_units",
    ]
    # Only keep columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols].reset_index(drop=True)

    out_path = os.path.join(out_dir, "01_raw_data.csv")
    df_out.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(df_out)} rows)")
    return out_path


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Fetch ChEMBL data")
    parser.add_argument("--receptor", required=True, help="Receptor/target name")
    parser.add_argument(
        "--bioactivity_type",
        default=None,
        help=f"Bioactivity type (default: {config.BIOACTIVITY_TYPE})",
    )
    args = parser.parse_args()
    run(args.receptor, args.bioactivity_type)
