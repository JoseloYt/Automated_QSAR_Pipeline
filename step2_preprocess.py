"""
Step 2 — Preprocess and clean the raw ChEMBL dataset.

Replicates the logic from preprocess(copia).ipynb:
  • Drop rows with missing SMILES / values
  • Remove duplicate SMILES (keep first)
  • Strip salts (keep longest fragment)
  • Validate & canonicalize SMILES with RDKit
  • Optionally convert IC50 → pIC50
  • Classify bioactivity

Usage:
    python step2_preprocess.py --receptor FGFR1
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

import config


def _strip_salts(smiles: str) -> str:
    """Split on '.' and keep the longest fragment (salt stripping)."""
    parts = str(smiles).split(".")
    return max(parts, key=len)


def _validate_and_canonicalize(smiles: str):
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

#Aqui hay que tener cuidado porque no siempre tengo nM
#Puede ser % de inhibición
#Esto cubre IC50 y KI
def _to_pic50(ic50_nm: float) -> float:
    """Convert IC50 in nM to pIC50 = -log10(IC50 in M)."""
    if ic50_nm <= 0:
        ic50_nm = 1 #Assuming 1nM is practically the lowest measurable value
    return -np.log10(ic50_nm * 1e-9)

#Cut values bigger than 100,000,000
def norm_value(input):
  norm = []

  for i in input['standard_value']:
    if i > 100000000:
      i = 10000000
    norm.append(i)

  input['standard_value_norm'] = norm
  x = input.drop('standard_value', axis=1)

  return x

def run(receptor_name: str, bioactivity_type: str | None = None) -> str:
    """
    Main entry point for Step 2.
    Returns path to the preprocessed CSV.
    """
    bio_type = bioactivity_type or config.BIOACTIVITY_TYPE
    out_dir = config.get_output_dir(receptor_name)
    raw_path = os.path.join(out_dir, "01_raw_data.csv")

    print(f"[Step 2] Preprocessing data for '{receptor_name}'")

    # ── Load ──────────────────────────────────────────────────
    df = pd.read_csv(raw_path)
    print(f"  Loaded {len(df)} rows from {raw_path}")

    # ── Drop missing ──────────────────────────────────────────
    df = df[df["canonical_smiles"].notna() & df["standard_value"].notna()].copy()
    print(f"  After dropping NaN SMILES/values: {len(df)}")

    # ── Remove duplicates (keep first occurrence) ─────────────
    df = df.drop_duplicates(subset=["canonical_smiles"]).copy()
    print(f"  After removing duplicate SMILES: {len(df)}")

    # ── Salt stripping ────────────────────────────────────────
    df["clean_smiles"] = df["canonical_smiles"].apply(_strip_salts)

    # ── Validate & canonicalize ───────────────────────────────
    df["clean_smiles"] = df["clean_smiles"].apply(_validate_and_canonicalize)
    invalid_count = df["clean_smiles"].isna().sum()
    if invalid_count > 0:
        print(f"  Dropping {invalid_count} invalid SMILES")
        df = df[df["clean_smiles"].notna()].copy()

    # ── Remove duplicates after canonicalization ──────────────
    df = df.drop_duplicates(subset=["clean_smiles"]).copy()
    print(f"  After canonicalization de-dup: {len(df)}")

    #Cut values bigger than 100,000,000
    df = norm_value(df)

    # ── Convert to pIC50 if applicable ────────────────────────
    convert = config.CONVERT_TO_PIC50 and bio_type.upper() in ("IC50", "KI")
    if convert:
        df["pIC50"] = df["standard_value_norm"].apply(_to_pic50)
        df = df[df["pIC50"].notna()].copy()
        target_col = "pIC50"
        print(f"  Converted to pIC50. Range: "
              f"{df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}")
    else:
        target_col = "standard_value"

    # ── Classify bioactivity ──────────────────────────────────
    if convert:
        conditions = [
            df[target_col] >= config.ACTIVE_THRESHOLD,
            df[target_col] < config.INACTIVE_THRESHOLD,
        ]
        choices = ["active", "inactive"]
        df["bioactivity_class"] = np.select(conditions, choices, default="intermediate")
    else:
        # For non-IC50 types, skip classification or use median split
        median_val = df[target_col].median()
        df["bioactivity_class"] = np.where(
            df[target_col] >= median_val, "high", "low"
        )

    # Print class distribution
    print(f"  Bioactivity class distribution:")
    for cls, cnt in df["bioactivity_class"].value_counts().items():
        print(f"    {cls}: {cnt}")

    # ── Select & save ─────────────────────────────────────────
    df_out = df[["clean_smiles", target_col, "bioactivity_class"]].copy()
    df_out.columns = ["Smiles", target_col, "bioactivity_class"]
    df_out = df_out.reset_index(drop=True)

    out_path = os.path.join(out_dir, "02_preprocessed.csv")
    df_out.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(df_out)} rows)")
    return out_path


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Preprocess data")
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--bioactivity_type", default=None)
    args = parser.parse_args()
    run(args.receptor, args.bioactivity_type)
