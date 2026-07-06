"""
Step 3 — Exploratory Data Analysis & Lipinski descriptors.

Replicates the logic from EDA_Lipinski(copia).ipynb:
  • Compute Lipinski descriptors (MW, LogP, NumHDonors, NumHAcceptors)
  • Drop molecules where RDKit fails
  • Generate publication-quality plots
  • Save enriched dataset

Usage:
    python step3_eda_lipinski.py --receptor FGFR1
"""

import argparse
import os
import warnings

import json 

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from scipy.stats import mannwhitneyu

import config

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(style="ticks")


# ── Lipinski Descriptor Computation ───────────────────────────
def compute_lipinski(smiles_list: list[str]) -> pd.DataFrame:
    """Compute Lipinski descriptors for a list of SMILES."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append([np.nan, np.nan, np.nan, np.nan])
            continue
        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
        ])
    return pd.DataFrame(rows, columns=["MW", "LogP", "NumHDonors", "NumHAcceptors"])


# ── Plotting Functions ────────────────────────────────────────
def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {path}")


def plot_target_distribution(df: pd.DataFrame, target_col: str, plots_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[target_col], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel(target_col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of {target_col}", fontsize=14)
    _save(fig, os.path.join(plots_dir, "target_distribution.png"))


def plot_class_frequency(df: pd.DataFrame, plots_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["bioactivity_class"].value_counts()
    counts.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Bioactivity Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Bioactivity Class Distribution", fontsize=14)
    plt.xticks(rotation=0)
    _save(fig, os.path.join(plots_dir, "class_frequency.png"))


def plot_lipinski_boxplots(df: pd.DataFrame, plots_dir: str):
    descriptors = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, desc in enumerate(descriptors):
        sns.boxplot(
            x="bioactivity_class", y=desc, data=df, ax=axes[i], palette="Set2"
        )
        axes[i].set_title(desc, fontsize=13)
        axes[i].set_xlabel("")
    fig.suptitle("Lipinski Descriptors by Bioactivity Class", fontsize=15, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(plots_dir, "lipinski_boxplots.png"))


def plot_chemical_space(df: pd.DataFrame, plots_dir: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    classes = df["bioactivity_class"].unique()
    palette = sns.color_palette("Set2", len(classes))
    for cls, color in zip(classes, palette):
        sub = df[df["bioactivity_class"] == cls]
        ax.scatter(sub["MW"], sub["LogP"], label=cls, alpha=0.5, s=15, c=[color])
    ax.set_xlabel("Molecular Weight", fontsize=12)
    ax.set_ylabel("LogP", fontsize=12)
    ax.set_title("Chemical Space (MW vs LogP)", fontsize=14)
    ax.legend()
    _save(fig, os.path.join(plots_dir, "chemical_space.png"))


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str, plots_dir: str):
    cols = [target_col, "MW", "LogP", "NumHDonors", "NumHAcceptors"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Descriptor Correlation Heatmap", fontsize=14)
    _save(fig, os.path.join(plots_dir, "correlation_heatmap.png"))

def mannwhitney_lipinski_tests(df: pd.DataFrame, activity_col: str = "pIC50") -> list[dict]:
    """Run Mann-Whitney U tests for Lipinski descriptors using active/inactive pIC50 classes."""
    descriptors = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    alpha = 0.05
    threshold = 6.5
    results = []

    df_classified = df.copy()
    df_classified["mannwhitney_class"] = np.where(
        df_classified[activity_col] >= threshold,
        "active",
        "inactive",
    )

    for descriptor in descriptors:
        active = df_classified.loc[
            df_classified["mannwhitney_class"] == "active", descriptor
        ].dropna()

        inactive = df_classified.loc[
            df_classified["mannwhitney_class"] == "inactive", descriptor
        ].dropna()

        if active.empty or inactive.empty:
            results.append({
                "Descriptor": descriptor,
                "Statistics": None,
                "p": None,
                "alpha": alpha,
                "threshold": threshold,
                "activity_col": activity_col,
                "Interpretation": (
                    f"Mann-Whitney U test could not be performed for {descriptor} "
                    "because one of the groups is empty."
                ),
                "n_active": int(len(active)),
                "n_inactive": int(len(inactive)),
            })
            continue

        stat, p = mannwhitneyu(active, inactive, alternative="two-sided")

        if p > alpha:
            interpretation = (
                f"The difference between active and inactive for {descriptor} "
                "is NOT statistically significant (fail to reject H0)."
            )
        else:
            interpretation = (
                f"The difference between active and inactive for {descriptor} "
                "IS statistically significant (reject H0)."
            )

        results.append({
            "Descriptor": descriptor,
            "Statistics": float(stat),
            "p": float(p),
            "alpha": alpha,
            "threshold": threshold,
            "activity_col": activity_col,
            "Interpretation": interpretation,
            "n_active": int(len(active)),
            "n_inactive": int(len(inactive)),
        })

    return results

# ── Main ──────────────────────────────────────────────────────
def run(receptor_name: str, bioactivity_type: str | None = None) -> str:
    """
    Main entry point for Step 3.
    Returns path to the enriched CSV with Lipinski descriptors.
    """
    bio_type = bioactivity_type or config.BIOACTIVITY_TYPE
    out_dir = config.get_output_dir(receptor_name)
    plots_dir = config.get_plots_dir(receptor_name)
    preproc_path = os.path.join(out_dir, "02_preprocessed.csv")

    print(f"[Step 3] EDA & Lipinski descriptors for '{receptor_name}'")

    df = pd.read_csv(preproc_path)
    print(f"  Loaded {len(df)} rows")

    # Determine target column
    convert = config.CONVERT_TO_PIC50 and bio_type.upper() in ("IC50", "KI")
    target_col = "pIC50" if convert else "standard_value"

    # Compute Lipinski descriptors
    print("  Computing Lipinski descriptors...")
    lip = compute_lipinski(df["Smiles"].tolist())
    df_lip = pd.concat([df, lip], axis=1)

    # Drop rows where Lipinski failed (invalid molecules)
    na_count = df_lip["MW"].isna().sum()
    if na_count > 0:
        print(f"  Dropping {na_count} molecules with invalid descriptors")
        df_lip = df_lip[df_lip["MW"].notna()].copy()

    df_lip = df_lip.reset_index(drop=True)
    print(f"  Final dataset: {len(df_lip)} rows")

        # Mann-Whitney U tests for Lipinski descriptors
    print("  Running Mann-Whitney U tests...")
    mw_results = mannwhitney_lipinski_tests(df_lip, activity_col=target_col)
    mw_out_path = os.path.join(out_dir, "03_mannwhitney_lipinski.json")
    with open(mw_out_path, "w", encoding="utf-8") as f:
        json.dump(mw_results, f, indent=2)
    print(f"  Mann-Whitney results -> {mw_out_path}")

    # Generate plots
    print("  Generating plots...")
    plot_target_distribution(df_lip, target_col, plots_dir)
    plot_class_frequency(df_lip, plots_dir)
    plot_lipinski_boxplots(df_lip, plots_dir)
    plot_chemical_space(df_lip, plots_dir)
    plot_correlation_heatmap(df_lip, target_col, plots_dir)

    # Save enriched dataset
    out_path = os.path.join(out_dir, "03_eda_lipinski.csv")
    df_lip.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: EDA & Lipinski")
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--bioactivity_type", default=None)
    args = parser.parse_args()
    run(args.receptor, args.bioactivity_type)
