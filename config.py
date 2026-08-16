"""
Central configuration for the automated QSAR pipeline.
All tunable parameters are defined here.
"""

import os

# ─────────────────────────────────────────────────────────────
# ChEMBL data acquisition settings
# ─────────────────────────────────────────────────────────────
BIOACTIVITY_TYPE = "IC50"          # e.g. "IC50", "Inhibition", "Ki", "EC50"
ORGANISM = "Homo sapiens"
TARGET_TYPE = "SINGLE PROTEIN"

# ─────────────────────────────────────────────────────────────
# Preprocessing settings
# ─────────────────────────────────────────────────────────────
# pIC50 conversion only applies when BIOACTIVITY_TYPE is "IC50" or "Ki"
CONVERT_TO_PIC50 = True
# Bioactivity classification thresholds (on pIC50 scale)
ACTIVE_THRESHOLD = 6.0       # pIC50 >= 6 → active
INACTIVE_THRESHOLD = 5.0     # pIC50 < 5  → inactive
                              # between 5 and 6 → intermediate

# ─────────────────────────────────────────────────────────────
# Morgan fingerprint settings (for Random Forest)
# ─────────────────────────────────────────────────────────────
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

# ─────────────────────────────────────────────────────────────
# Train / test split settings
# ─────────────────────────────────────────────────────────────
RF_TEST_SIZE = 0.2            # 80/20 for Random Forest
RF_RANDOM_STATE = 42

CHEMPROP_SPLIT = (0.8, 0.2)   # 80/20 (train/val) for Chemprop
CHEMPROP_SPLIT_TYPE = "random"

# ─────────────────────────────────────────────────────────────
# Chemprop training settings
# ─────────────────────────────────────────────────────────────
CHEMPROP_MAX_EPOCHS = 50
CHEMPROP_BATCH_SIZE = 64
CHEMPROP_NUM_WORKERS = 0

# ─────────────────────────────────────────────────────────────
# Applicability domain (Tanimoto AD10) settings
# ─────────────────────────────────────────────────────────────
AD10_K = 10                   # number of nearest neighbors
AD10_PERCENTILE = 5           # low-percentile cutoff of training AD10 distribution

# ─────────────────────────────────────────────────────────────
# Conformal prediction settings
# ─────────────────────────────────────────────────────────────
CP_ALPHA = 0.05                # -> 95% target coverage
CP_MIN_STRATUM_SIZE = 20       # min calibration points per Mondrian (AD) stratum
CP_COVERAGE_SWEEP_ALPHAS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
# Error threshold (in target-column units) used only for the AD x reliability
# heatmap diagnostic plot; does not affect AD or CP calibration itself.
AD_RELIABILITY_ERROR_THRESHOLD = 1.0  # e.g. 1 log unit of pIC50

# ─────────────────────────────────────────────────────────────
# Output directory helpers
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def get_output_dir(receptor_name: str) -> str:
    """Return (and create) the output directory for a given receptor."""
    out = os.path.join(BASE_OUTPUT_DIR, receptor_name)
    os.makedirs(out, exist_ok=True)
    return out


def get_models_dir(receptor_name: str) -> str:
    """Return (and create) the models sub-directory."""
    d = os.path.join(get_output_dir(receptor_name), "models")
    os.makedirs(d, exist_ok=True)
    return d


def get_plots_dir(receptor_name: str) -> str:
    """Return (and create) the plots sub-directory."""
    d = os.path.join(get_output_dir(receptor_name), "plots")
    os.makedirs(d, exist_ok=True)
    return d
