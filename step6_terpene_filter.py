"""
step6_terpene_filter.py
=======================

Terpene / terpenoid / isoprenoid / natural-product enrichment of the QSAR
datasets, so that models retrained on the reduced set have an applicability
domain that actually covers the (sesqui)terpene molecules we want to predict.

Motivation
----------
The ChEMBL datasets for kinase targets (FGFR*, EGFR, PDGFR*, TGFBR*) are
dominated by synthetic, flat, N-rich, aromatic inhibitors. Essential-oil
compounds -- mostly mono- and sesquiterpenes -- are small, sp3-rich,
hydrocarbon-like molecules that sit far outside that chemical space, so
*every* prediction lands outside the AD10 applicability domain and the
conformal intervals blow up.

This script scores every training molecule for how "terpene-like" and how
"natural-product-like" it is, and selects the sub-population that shares
chemistry with the candidate set. The output is a smaller, chemically
coherent training set whose AD is much more likely to contain the candidates.

What it computes per molecule
-----------------------------
1. NP-likeness score (Ertl et al., J. Chem. Inf. Model. 2008, 48, 68-74)
   via the RDKit NP_Score contrib model. Range roughly [-5, +5];
   > 0 = natural-product-like, < 0 = synthetic-like.
2. Isoprene-rule descriptors: carbon count, C % 5 residual, degree of
   unsaturation, terpene class assignment by carbon skeleton size
   (hemi/mono/sesqui/di/sester/tri/tetra-terpene).
3. Terpene / isoprenoid substructure hits (SMARTS): isoprene unit,
   isopropyl / isopropenyl / gem-dimethyl, head-to-tail isoprenoid chain,
   methyl-on-ring, cyclohexane / cyclopentane fused aliphatic systems,
   common terpenoid oxidation patterns.
4. Shape / composition descriptors: Fsp3, aromatic ring count, heteroatom
   fraction, N count, halogen count, C/O ratio, Bertz complexity, ring count.
5. Anti-patterns that disqualify a kinase inhibitor from being terpene-like:
   aromatic N-heterocycles, sulfonamides, amides/anilines, halogens, etc.
6. Tanimoto similarity to the candidate set (max and mean-of-top-5), if a
   candidate file is supplied.

Composite selection
-------------------
`terpene_score` in [0, 1] is a weighted blend of the normalised NP score,
the isoprenoid/skeleton evidence, the sp3/aliphatic character and (optionally)
the similarity to the candidates. Molecules above `--score-cutoff` are kept;
if fewer than `--min-n` pass, the top `--min-n` by score are kept anyway so
that a model can still be trained (this is reported explicitly, never silent).

Outputs (per receptor, in output/<RECEPTOR>/)
---------------------------------------------
  06_terpene_annotated.csv   every molecule + all descriptors + terpene_score
  06_terpene_subset.csv      the reduced training set (same columns as input)
  06_terpene_filter.json     thresholds used, counts at each rule, AD report
  plots/06_np_score_distribution.png
  plots/06_terpene_score_vs_activity.png
  plots/06_ad_coverage_full_vs_subset.png
  plots/06_chemical_space_pca.png

And globally, in output/:
  06_terpene_summary.csv     one row per receptor
  06_terpene_summary.json
  06_terpene_summary_plots.png

Usage
-----
    # all receptors found under output/
    python step6_terpene_filter.py --all

    # one receptor
    python step6_terpene_filter.py --receptor FGFR1

    # anchoring the selection on the molecules you actually want to predict
    python step6_terpene_filter.py --all \
        --candidates "output/FGFR1/predictions/All_EO_compounds_SMILES_predictions.csv"

    # more permissive / more aggressive
    python step6_terpene_filter.py --all --score-cutoff 0.35 --min-n 400

After running, retrain on the subset by pointing step4/step5 at
06_terpene_subset.csv (see --emit-retrain-cmd).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")

# Local pipeline modules (optional -- the script also runs standalone)
try:
    import config as pipeline_config
except Exception:  # pragma: no cover
    pipeline_config = None

logger = logging.getLogger("step6_terpene_filter")


# ═════════════════════════════════════════════════════════════
# Defaults (overridable from the CLI, and from config.py if present)
# ═════════════════════════════════════════════════════════════
DEFAULT_SCORE_CUTOFF = 0.45
DEFAULT_MIN_N = 300
DEFAULT_MORGAN_RADIUS = getattr(pipeline_config, "MORGAN_RADIUS", 2)
DEFAULT_MORGAN_NBITS = getattr(pipeline_config, "MORGAN_NBITS", 2048)
DEFAULT_AD10_K = getattr(pipeline_config, "AD10_K", 10)
DEFAULT_AD10_PERCENTILE = getattr(pipeline_config, "AD10_PERCENTILE", 5)

INPUT_CANDIDATES = ["03_eda_lipinski.csv", "02_preprocessed.csv"]
SMILES_COLUMNS = ["Smiles", "SMILES", "Clean SMILES", "canonical_smiles", "smiles"]
TARGET_COLUMNS = ["pIC50", "pKi", "standard_value", "target"]

# Composite score weights -- exposed so they can be tuned in one place.
WEIGHTS = {
    "np_score": 0.25,        # Ertl natural-product likeness
    "isoprenoid": 0.25,      # isoprene rule + terpene SMARTS evidence
    "aliphatic": 0.25,       # Fsp3 / non-aromatic / low heteroatom character
    "similarity": 0.25,      # Tanimoto to the candidate set (redistributed if absent)
}


# ═════════════════════════════════════════════════════════════
# SMARTS library
# ═════════════════════════════════════════════════════════════
def _smarts(d):
    out = {}
    for name, patt in d.items():
        m = Chem.MolFromSmarts(patt)
        if m is None:
            logger.warning("Invalid SMARTS for %s: %s", name, patt)
        else:
            out[name] = m
    return out


# Positive evidence: isoprenoid / terpene structural motifs
TERPENE_SMARTS = _smarts({
    # C5 isoprene skeleton (2-methylbuta-1,3-diene) and its saturated form
    "isoprene_unit":        "[CX4,CX3;!R]~[CX4,CX3](~[CH3])~[CX4,CX3]~[CX4,CX3]",
    "isoprene_diene":       "C=C(C)C=C",
    # head-to-tail prenyl / isoprenoid chain
    "prenyl":               "[CH3][C]([CH3])=[CH][CH2]",
    "prenyl_chain":         "[CH2][CH]=[C]([CH3])[CH2][CH2][CH]=[C]([CH3])",
    "isopropenyl":          "[CH2]=[C]([CH3])[CX4]",
    "isopropyl":            "[CH3][CH]([CH3])[CX4,CX3]",
    "gem_dimethyl":         "[CX4]([CH3])([CH3])",
    # methyls decorating an aliphatic ring -- ubiquitous in terpenes
    "methyl_on_ring":       "[CH3][CX4;R]",
    "methyl_on_ring_sp2":   "[CH3][CX3;R]=[CX3;R]",
    # bridged / fused carbocycles typical of mono- and sesquiterpenes
    "gem_dimethyl_ring":    "[CX4;R]([CH3])([CH3])",
    "cyclohexane_aliph":    "[CX4;R1]1[CX4,CX3;R1][CX4,CX3;R1][CX4,CX3;R1][CX4,CX3;R1][CX4,CX3;R1]1",
    "cyclopentane_aliph":   "[CX4;R]1[CX4,CX3;R][CX4,CX3;R][CX4,CX3;R][CX4,CX3;R]1",
    "bicyclic_bridge":      "[CX4;R2]",
    # terpenoid oxidation patterns
    "allylic_alcohol":      "[OX2H][CX4][CX3]=[CX3]",
    "tertiary_alcohol":     "[OX2H][CX4]([#6])([#6])[#6]",
    "enone":                "[CX3]=[CX3][CX3]=[OX1]",
    "cyclic_ether_ox":      "[OX2;R]([CX4;R])[CX4;R]",
    "lactone":              "[CX3](=[OX1])[OX2;R]",
    "epoxide":              "[CX4;R1]1[OX2;R1][CX4;R1]1",
    "exocyclic_methylene":  "[CH2]=[CX3;R]",
    # alpha-methylene-gamma-lactone: the sesquiterpene-lactone warhead
    "alpha_methylene_lactone": "C=C1CCOC1=O",
})

# Negative evidence: hallmarks of synthetic kinase inhibitors
ANTI_TERPENE_SMARTS = _smarts({
    "aromatic_n_hetero":  "[n]",
    "sulfonamide":        "[SX4](=[OX1])(=[OX1])[NX3]",
    "sulfone":            "[SX4](=[OX1])(=[OX1])",
    "anilide":            "[NX3][CX3](=[OX1])",
    "aryl_halide":        "[c][F,Cl,Br,I]",
    "trifluoromethyl":    "[CX4]([F])([F])[F]",
    "nitrile":            "[CX2]#[NX1]",
    "nitro":              "[NX3](=[OX1])=[OX1]",
    "biaryl":             "[c][c;!R1]",
    "urea":               "[NX3][CX3](=[OX1])[NX3]",
    "phosphate":          "[PX4](=[OX1])",
    "quinazoline":        "c1ccc2ncncc2c1",
})

# Terpene classes by carbon-skeleton size (allowing decoration)
TERPENE_CLASSES = [
    ("hemiterpene",  5,  5),
    ("monoterpene", 10, 10),
    ("sesquiterpene", 15, 15),
    ("diterpene",   20, 20),
    ("sesterterpene", 25, 25),
    ("triterpene",  30, 30),
    ("tetraterpene", 40, 40),
]
CARBON_TOLERANCE = 2  # allow +/- 2 C for demethylation / small side chains


# ═════════════════════════════════════════════════════════════
# NP-likeness scorer
# ═════════════════════════════════════════════════════════════
class NPScorer:
    """Thin wrapper over the RDKit NP_Score contrib, with a graceful fallback."""

    def __init__(self):
        self.fscore = None
        self.available = False
        try:
            from rdkit.Chem import RDConfig
            np_dir = os.path.join(RDConfig.RDContribDir, "NP_Score")
            if np_dir not in sys.path:
                sys.path.append(np_dir)
            import npscorer  # type: ignore
            self.fscore = npscorer.readNPModel()
            self._scorer = npscorer
            self.available = True
            logger.info("NP-likeness model loaded from %s", np_dir)
        except Exception as exc:
            logger.warning(
                "NP_Score contrib not available (%s). "
                "Falling back to a surrogate NP-likeness estimate based on "
                "Fsp3 / aromaticity / heteroatom composition. "
                "Install a full RDKit (conda-forge rdkit, or pip rdkit>=2022.09) "
                "to get the real Ertl score.", exc,
            )

    def score(self, mol) -> float:
        if self.available:
            try:
                return float(self._scorer.scoreMol(mol, self.fscore))
            except Exception:
                return np.nan
        return self._surrogate(mol)

    @staticmethod
    def _surrogate(mol) -> float:
        """
        Crude stand-in for the Ertl score, mapped onto a comparable [-5, 5]
        range: rewards sp3 character and oxygen-only heteroatom composition,
        penalises aromatic rings, nitrogen and halogens.
        """
        try:
            heavy = mol.GetNumHeavyAtoms() or 1
            fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
            n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
            n_N = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
            n_hal = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
            n_O = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
            s = (4.0 * fsp3) - (1.4 * n_arom) - (0.9 * n_N / max(1, heavy) * heavy * 0.25) \
                - (1.2 * n_hal) + (0.15 * n_O)
            return float(np.clip(s, -5.0, 5.0))
        except Exception:
            return np.nan


# ═════════════════════════════════════════════════════════════
# Per-molecule descriptor computation
# ═════════════════════════════════════════════════════════════
def classify_terpene_skeleton(n_carbons: int) -> tuple[str, bool, int]:
    """
    Assign a terpene class from the carbon count.
    Returns (class_name, obeys_isoprene_rule, carbon_deviation).
    """
    best = ("non-terpene-skeleton", False, 99)
    for name, lo, hi in TERPENE_CLASSES:
        dev = min(abs(n_carbons - lo), abs(n_carbons - hi))
        if dev <= CARBON_TOLERANCE and dev < best[2]:
            best = (name, dev == 0, dev)
    return best


def describe_molecule(smiles: str, np_scorer: NPScorer) -> dict | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    atoms = list(mol.GetAtoms())
    heavy = mol.GetNumHeavyAtoms()
    n_C = sum(1 for a in atoms if a.GetSymbol() == "C")
    n_O = sum(1 for a in atoms if a.GetSymbol() == "O")
    n_N = sum(1 for a in atoms if a.GetSymbol() == "N")
    n_S = sum(1 for a in atoms if a.GetSymbol() == "S")
    n_hal = sum(1 for a in atoms if a.GetSymbol() in ("F", "Cl", "Br", "I"))
    n_hetero = heavy - n_C

    d: dict = {}
    d["n_heavy_atoms"] = heavy
    d["n_carbons"] = n_C
    d["n_oxygens"] = n_O
    d["n_nitrogens"] = n_N
    d["n_sulfurs"] = n_S
    d["n_halogens"] = n_hal
    d["heteroatom_fraction"] = n_hetero / heavy if heavy else np.nan
    d["C_over_O"] = n_C / n_O if n_O else np.inf

    # --- shape / hybridisation ---
    d["Fsp3"] = rdMolDescriptors.CalcFractionCSP3(mol)
    d["n_rings"] = rdMolDescriptors.CalcNumRings(mol)
    d["n_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    d["n_aliphatic_rings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    d["n_aromatic_atoms"] = sum(1 for a in atoms if a.GetIsAromatic())
    d["aromatic_atom_fraction"] = d["n_aromatic_atoms"] / heavy if heavy else np.nan
    d["n_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    d["n_stereocenters"] = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    d["n_bridgehead_atoms"] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    d["n_spiro_atoms"] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    d["bertz_complexity"] = Descriptors.BertzCT(mol)
    d["MW_calc"] = Descriptors.MolWt(mol)
    d["LogP_calc"] = Crippen.MolLogP(mol)
    d["TPSA"] = rdMolDescriptors.CalcTPSA(mol)

    # --- isoprene rule ---
    d["carbons_mod5"] = n_C % 5
    d["obeys_C5_rule"] = bool(n_C >= 5 and n_C % 5 == 0)
    tclass, exact, dev = classify_terpene_skeleton(n_C)
    # A carbon count alone is meaningless -- almost any drug-like molecule has
    # 15-25 carbons. Only credit the skeleton when the molecule is also
    # hydrocarbon-like enough for the isoprene rule to mean anything.
    skeleton_plausible = bool(
        tclass != "non-terpene-skeleton"
        and d["n_aromatic_rings"] <= 1
        and d["Fsp3"] >= 0.35
        and n_hetero <= max(1, int(0.30 * heavy))
    )
    if not skeleton_plausible:
        tclass, exact, dev = "non-terpene-skeleton", False, 99
    d["terpene_class"] = tclass
    d["terpene_class_exact"] = exact
    d["carbon_deviation"] = dev if dev < 99 else np.nan
    d["is_sesquiterpene_size"] = bool(tclass == "sesquiterpene")
    d["is_mono_or_sesqui"] = bool(tclass in ("monoterpene", "sesquiterpene"))

    # --- SMARTS evidence ---
    pos_hits = 0
    for name, patt in TERPENE_SMARTS.items():
        n = len(mol.GetSubstructMatches(patt))
        d[f"smarts_{name}"] = n
        if n > 0:
            pos_hits += 1
    d["n_terpene_smarts_hit"] = pos_hits

    neg_hits = 0
    for name, patt in ANTI_TERPENE_SMARTS.items():
        n = len(mol.GetSubstructMatches(patt))
        d[f"anti_{name}"] = n
        if n > 0:
            neg_hits += 1
    d["n_anti_smarts_hit"] = neg_hits

    # A hard, interpretable boolean: "this really looks like a terpenoid"
    d["is_terpenoid_strict"] = bool(
        d["n_aromatic_rings"] == 0
        and n_N == 0
        and n_hal == 0
        and n_S == 0
        and d["Fsp3"] >= 0.6
        and n_C >= 8
        and (d["C_over_O"] >= 3 if n_O else True)
        and (
            d["smarts_isopropyl"] or d["smarts_isopropenyl"]
            or d["smarts_gem_dimethyl"] or d["smarts_prenyl"]
            or d["smarts_methyl_on_ring"] or d["smarts_methyl_on_ring_sp2"]
        )
    )

    # --- NP likeness ---
    d["NP_score"] = np_scorer.score(mol)

    return d


# ═════════════════════════════════════════════════════════════
# Sub-score construction
# ═════════════════════════════════════════════════════════════
def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def compute_subscores(df: pd.DataFrame) -> pd.DataFrame:
    """Build the three intrinsic sub-scores, each in [0, 1]."""
    out = pd.DataFrame(index=df.index)

    # 1. NP-likeness, mapped from roughly [-3, +3] onto [0, 1]
    np_s = df["NP_score"].astype(float)
    out["sub_np"] = _clip01((np_s + 3.0) / 6.0).fillna(0.0)

    # 2. Isoprenoid / skeleton evidence
    iso = np.zeros(len(df), dtype=float)
    iso += 0.30 * df["terpene_class"].ne("non-terpene-skeleton").astype(float)
    iso += 0.10 * df["terpene_class_exact"].astype(float)
    iso += 0.10 * df["obeys_C5_rule"].astype(float)
    # breadth of terpene SMARTS evidence (saturates at ~6 distinct motifs)
    iso += 0.30 * _clip01(df["n_terpene_smarts_hit"] / 6.0)
    # specific strong markers
    strong = (
        df["smarts_isopropenyl"].gt(0)
        | df["smarts_prenyl"].gt(0)
        | df["smarts_prenyl_chain"].gt(0)
        | df["smarts_alpha_methylene_lactone"].gt(0)
    )
    iso += 0.20 * strong.astype(float)
    out["sub_isoprenoid"] = _clip01(iso)

    # 3. Aliphatic / natural-product-shaped character, penalised by
    #    synthetic-inhibitor hallmarks
    ali = np.zeros(len(df), dtype=float)
    ali += 0.45 * _clip01(df["Fsp3"].astype(float))
    ali += 0.20 * _clip01(1.0 - df["aromatic_atom_fraction"].astype(float).fillna(1.0))
    ali += 0.15 * _clip01(1.0 - df["heteroatom_fraction"].astype(float).fillna(1.0) / 0.4)
    ali += 0.10 * (df["n_nitrogens"] == 0).astype(float)
    ali += 0.10 * (df["n_halogens"] == 0).astype(float)
    # penalty: each distinct anti-pattern family costs 8%, capped at 40%
    ali -= np.minimum(0.08 * df["n_anti_smarts_hit"].astype(float), 0.40)
    out["sub_aliphatic"] = _clip01(ali)

    return out


def compute_similarity_subscore(
    df_fps: np.ndarray, cand_fps: np.ndarray, top_k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tanimoto similarity of each dataset molecule to the candidate set.
    Returns (max_similarity, mean_of_top_k_similarity).
    """
    q = np.asarray(df_fps, dtype=np.float32)
    r = np.asarray(cand_fps, dtype=np.float32)
    dot = q @ r.T
    union = q.sum(1, dtype=np.float32)[:, None] + r.sum(1, dtype=np.float32)[None, :] - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, dot / union, 0.0)
    k = min(top_k, sim.shape[1])
    topk = np.partition(sim, -k, axis=1)[:, -k:]
    return sim.max(axis=1).astype(np.float64), topk.mean(axis=1).astype(np.float64)


def combine_score(sub: pd.DataFrame, has_similarity: bool) -> pd.Series:
    w = dict(WEIGHTS)
    if not has_similarity:
        # redistribute the similarity weight over the three intrinsic terms
        extra = w.pop("similarity")
        total = sum(w.values())
        for k in w:
            w[k] += extra * w[k] / total
    score = (
        w["np_score"] * sub["sub_np"]
        + w["isoprenoid"] * sub["sub_isoprenoid"]
        + w["aliphatic"] * sub["sub_aliphatic"]
    )
    if has_similarity:
        score = score + w["similarity"] * sub["sub_similarity"]
    return score.clip(0.0, 1.0)


# ═════════════════════════════════════════════════════════════
# Fingerprints / AD
# ═════════════════════════════════════════════════════════════
def morgan_matrix(smiles_list, radius: int, n_bits: int):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.int8)
    valid = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        fps[i] = arr
        valid[i] = True
    return fps, valid


def ad10_scores(query_fps, ref_fps, k=10, exclude_self=False, chunk=1024):
    """
    Mean Tanimoto similarity to the k nearest neighbours in ref_fps.

    Computed in row chunks with float32 so that datasets of 10k+ molecules
    (e.g. EGFR) don't allocate an n x n float64 similarity matrix.
    """
    q = np.asarray(query_fps, dtype=np.float32)
    r = np.asarray(ref_fps, dtype=np.float32)
    n_ref = r.shape[0]
    k_eff = min(k, n_ref - 1 if exclude_self else n_ref)
    if k_eff <= 0 or q.shape[0] == 0:
        return np.zeros(q.shape[0])

    r_sum = r.sum(1, dtype=np.float32)[None, :]
    out = np.empty(q.shape[0], dtype=np.float64)
    for start in range(0, q.shape[0], chunk):
        stop = min(start + chunk, q.shape[0])
        qi = q[start:stop]
        dot = qi @ r.T
        union = qi.sum(1, dtype=np.float32)[:, None] + r_sum - dot
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, dot / union, 0.0)
        if exclude_self:
            rows = np.arange(stop - start)
            sim[rows, np.arange(start, stop)] = -1.0
        out[start:stop] = np.partition(sim, -k_eff, axis=1)[:, -k_eff:].mean(axis=1)
    return out


def ad_coverage_report(train_fps, cand_fps, k, percentile, label):
    """AD10 threshold from a training set, and how many candidates fall inside."""
    if train_fps.shape[0] < k + 1 or cand_fps.shape[0] == 0:
        return None
    train_ad = ad10_scores(train_fps, train_fps, k=k, exclude_self=True)
    thr = float(np.percentile(train_ad, percentile))
    cand_ad = ad10_scores(cand_fps, train_fps, k=k)
    inside = cand_ad >= thr
    return {
        "label": label,
        "n_train": int(train_fps.shape[0]),
        "ad10_threshold": thr,
        "train_ad10_mean": float(train_ad.mean()),
        "n_candidates": int(cand_fps.shape[0]),
        "n_candidates_in_AD": int(inside.sum()),
        "pct_candidates_in_AD": float(100.0 * inside.mean()),
        "candidate_ad10_mean": float(cand_ad.mean()),
        "candidate_ad10_max": float(cand_ad.max()),
        "_cand_ad10": cand_ad,
        "_train_ad10": train_ad,
    }


# ═════════════════════════════════════════════════════════════
# I/O helpers
# ═════════════════════════════════════════════════════════════
def find_column(df: pd.DataFrame, options) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in options:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_receptor_dataset(receptor_dir: str):
    for fname in INPUT_CANDIDATES:
        path = os.path.join(receptor_dir, fname)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            col = find_column(df, SMILES_COLUMNS)
            if col is None:
                logger.warning("No SMILES column in %s (columns: %s)", path, list(df.columns))
                continue
            return df, col, fname
    return None, None, None


def load_candidates(path: str):
    df = pd.read_csv(path)
    col = find_column(df, ["Clean SMILES", "Canonical SMILES", "SMILES", "Smiles", "smiles"])
    if col is None:
        raise ValueError(f"No SMILES column found in candidates file {path}")
    name_col = find_column(df, ["Molecule", "Name", "name", "compound"])
    smis, names = [], []
    for i, s in enumerate(df[col].tolist()):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        smis.append(Chem.MolToSmiles(mol))
        names.append(str(df[name_col].iloc[i]) if name_col else f"cand_{i}")
    return smis, names


# ═════════════════════════════════════════════════════════════
# Plots
# ═════════════════════════════════════════════════════════════
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_receptor(receptor, ann, subset_mask, cov_full, cov_sub, cand_np, plots_dir):
    plt = _plt()
    os.makedirs(plots_dir, exist_ok=True)

    # 1. NP score distribution: dataset vs kept subset vs candidates
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ann["NP_score"].dropna(), bins=50, alpha=0.55,
            label=f"full dataset (n={len(ann)})", color="#8899aa", density=True)
    ax.hist(ann.loc[subset_mask, "NP_score"].dropna(), bins=40, alpha=0.65,
            label=f"selected subset (n={int(subset_mask.sum())})", color="#2e8b57", density=True)
    if cand_np is not None and len(cand_np):
        ax.hist(cand_np, bins=25, alpha=0.65,
                label=f"candidates (n={len(cand_np)})", color="#c0392b", density=True)
    ax.axvline(0, ls="--", c="k", lw=1)
    ax.set_xlabel("NP-likeness score (Ertl)")
    ax.set_ylabel("density")
    ax.set_title(f"{receptor} — natural-product likeness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "06_np_score_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. terpene score vs activity
    tgt = find_column(ann, TARGET_COLUMNS)
    if tgt:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(ann["terpene_score"], ann[tgt], s=8, alpha=0.35,
                   c=np.where(subset_mask, "#2e8b57", "#99a3ad"))
        ax.set_xlabel("terpene_score")
        ax.set_ylabel(tgt)
        ax.set_title(f"{receptor} — terpene-likeness vs {tgt}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "06_terpene_score_vs_activity.png"), dpi=150)
        plt.close(fig)

    # 3. AD coverage before/after
    if cov_full and cov_sub:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, cov, title in zip(axes, [cov_full, cov_sub],
                                  ["Full dataset", "Terpene-selected subset"]):
            ax.hist(cov["_train_ad10"], bins=40, alpha=0.6, color="#8899aa",
                    label="training AD10", density=True)
            ax.hist(cov["_cand_ad10"], bins=20, alpha=0.7, color="#c0392b",
                    label="candidates AD10", density=True)
            ax.axvline(cov["ad10_threshold"], ls="--", c="k",
                       label=f"AD cutoff = {cov['ad10_threshold']:.3f}")
            ax.set_title(f"{title}\n{cov['n_candidates_in_AD']}/{cov['n_candidates']} "
                         f"candidates in AD ({cov['pct_candidates_in_AD']:.0f}%)")
            ax.set_xlabel("AD10 (mean Tanimoto to 10-NN)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("density")
        fig.suptitle(f"{receptor} — applicability-domain coverage of the candidates")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "06_ad_coverage_full_vs_subset.png"), dpi=150)
        plt.close(fig)


def plot_chemical_space(receptor, fps, subset_mask, cand_fps, plots_dir, max_points=3000):
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return
    plt = _plt()
    # Subsample the training points for the projection: PCA on a 10k x 2048
    # dense matrix is slow and the scatter is unreadable anyway. Selected
    # molecules are always kept, dropped ones are sampled.
    idx = np.arange(fps.shape[0])
    if fps.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = idx[subset_mask]
        rest = idx[~subset_mask]
        n_rest = max(0, max_points - len(keep))
        if n_rest < len(rest):
            rest = rng.choice(rest, size=n_rest, replace=False)
        idx = np.sort(np.concatenate([keep, rest]))
        fps = fps[idx]
        subset_mask = subset_mask[idx]

    n_train = fps.shape[0]
    allfp = np.vstack([fps, cand_fps]) if cand_fps is not None and len(cand_fps) else fps
    if allfp.shape[0] < 3:
        return
    coords = PCA(n_components=2, random_state=0).fit_transform(allfp.astype(np.float32))
    tr, cd = coords[:n_train], coords[n_train:]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(tr[~subset_mask, 0], tr[~subset_mask, 1], s=7, alpha=0.25,
               c="#99a3ad", label="dataset (dropped)")
    ax.scatter(tr[subset_mask, 0], tr[subset_mask, 1], s=12, alpha=0.7,
               c="#2e8b57", label="dataset (kept)")
    if len(cd):
        ax.scatter(cd[:, 0], cd[:, 1], s=45, marker="^", c="#c0392b",
                   edgecolors="k", linewidths=0.4, label="candidates")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{receptor} — chemical space (Morgan FP PCA)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "06_chemical_space_pca.png"), dpi=150)
    plt.close(fig)


def plot_summary(summary_df, out_path):
    plt = _plt()
    d = summary_df.sort_values("pct_candidates_in_AD_subset", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, max(4.5, 0.45 * len(d) + 2)))

    y = np.arange(len(d))
    axes[0].barh(y - 0.2, d["n_molecules"], height=0.4, color="#8899aa", label="full")
    axes[0].barh(y + 0.2, d["n_selected"], height=0.4, color="#2e8b57", label="selected")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d["receptor"], fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("molecules (log)")
    axes[0].set_title("Dataset size before / after")
    axes[0].legend(fontsize=8)

    axes[1].barh(y, d["mean_NP_score_selected"], color="#2e8b57", alpha=0.8, label="selected")
    axes[1].barh(y, d["mean_NP_score_full"], color="#8899aa", alpha=0.5, height=0.5, label="full")
    axes[1].axvline(0, ls="--", c="k", lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("mean NP-likeness")
    axes[1].set_title("Natural-product likeness")
    axes[1].legend(fontsize=8)

    axes[2].barh(y - 0.2, d["pct_candidates_in_AD_full"], height=0.4,
                 color="#8899aa", label="full")
    axes[2].barh(y + 0.2, d["pct_candidates_in_AD_subset"], height=0.4,
                 color="#c0392b", label="subset")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("% candidates inside AD")
    axes[2].set_title("Applicability-domain coverage")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════
# Per-receptor driver
# ═════════════════════════════════════════════════════════════
def process_receptor(
    receptor: str,
    base_output_dir: str,
    np_scorer: NPScorer,
    cand_smiles,
    cand_names,
    args,
) -> dict | None:
    receptor_dir = os.path.join(base_output_dir, receptor)
    df, smi_col, src = load_receptor_dataset(receptor_dir)
    if df is None:
        logger.warning("[%s] no usable dataset found — skipping", receptor)
        return None

    logger.info("[%s] %s: %d molecules (SMILES column '%s')", receptor, src, len(df), smi_col)

    # ---- descriptors ----
    records, keep_idx = [], []
    for i, smi in enumerate(df[smi_col].astype(str)):
        rec = describe_molecule(smi, np_scorer)
        if rec is None:
            continue
        records.append(rec)
        keep_idx.append(i)
    if not records:
        logger.warning("[%s] no parsable molecules — skipping", receptor)
        return None

    desc = pd.DataFrame(records, index=df.index[keep_idx])
    ann = pd.concat([df.loc[df.index[keep_idx]], desc], axis=1)
    n_unparsable = len(df) - len(ann)
    if n_unparsable:
        logger.info("[%s] %d SMILES could not be parsed by RDKit and were dropped",
                    receptor, n_unparsable)

    # ---- fingerprints ----
    fps, valid = morgan_matrix(ann[smi_col].astype(str).tolist(),
                               args.morgan_radius, args.morgan_nbits)
    ann = ann[valid].reset_index(drop=True)
    fps = fps[valid]

    # ---- sub-scores ----
    sub = compute_subscores(ann)
    cand_fps = np.zeros((0, args.morgan_nbits), dtype=np.int8)
    has_sim = False
    if cand_smiles:
        cand_fps, cvalid = morgan_matrix(cand_smiles, args.morgan_radius, args.morgan_nbits)
        cand_fps = cand_fps[cvalid]
        if len(cand_fps):
            sim_max, sim_top5 = compute_similarity_subscore(fps, cand_fps)
            ann["sim_max_to_candidates"] = sim_max
            ann["sim_top5_to_candidates"] = sim_top5
            # Tanimoto values against terpenes are intrinsically low; rescale
            # so that 0.35 max-similarity already counts as a strong match.
            sub["sub_similarity"] = _clip01(sim_top5 / 0.35)
            has_sim = True

    ann = pd.concat([ann, sub], axis=1)
    ann["terpene_score"] = combine_score(sub, has_sim)

    # ---- optional: optimise the subset size for AD coverage ----
    sweep_rows = []
    if args.optimize_n and len(cand_fps):
        order = ann["terpene_score"].to_numpy().argsort()[::-1]
        sizes = [n for n in args.sweep_sizes if n <= len(ann)]
        if len(ann) not in sizes:
            sizes.append(len(ann))
        for n in sizes:
            m = np.zeros(len(ann), dtype=bool)
            m[order[:n]] = True
            c = ad_coverage_report(fps[m], cand_fps, args.ad10_k,
                                   args.ad10_percentile, f"n={n}")
            if c is None:
                continue
            sweep_rows.append({
                "n": n,
                "ad10_threshold": c["ad10_threshold"],
                "n_candidates_in_AD": c["n_candidates_in_AD"],
                "pct_candidates_in_AD": c["pct_candidates_in_AD"],
                "candidate_ad10_mean": c["candidate_ad10_mean"],
            })
        if sweep_rows:
            sweep_str = ", ".join(
                f"{r['n']}:{r['pct_candidates_in_AD']:.0f}%" for r in sweep_rows)
            # Only sizes that would still give a trainable model are eligible.
            eligible = [r for r in sweep_rows if r["n"] >= args.min_trainable_n]
            # Best coverage; ties broken toward the LARGER subset (more data
            # to train on for the same AD coverage).
            best = max(eligible, key=lambda r: (r["pct_candidates_in_AD"], r["n"])) \
                if eligible else None
            best_any = max(sweep_rows, key=lambda r: r["pct_candidates_in_AD"])

            if best is not None and best["pct_candidates_in_AD"] > 0:
                logger.info("[%s] size sweep: %s -> picked n=%d (%.0f%% candidates in AD)",
                            receptor, sweep_str, best["n"], best["pct_candidates_in_AD"])
                # force the exact top-N selection
                args = argparse.Namespace(**{**vars(args), "min_n": best["n"],
                                             "score_cutoff": 1.1})
            else:
                # No trainable size gets any candidate inside the AD. Keeping
                # the whole dataset would be the degenerate tie-break, so fall
                # back to the requested min_n instead.
                logger.info("[%s] size sweep: %s", receptor, sweep_str)
                note = ""
                if best_any["pct_candidates_in_AD"] > 0:
                    note = (f" (n={best_any['n']} would reach "
                            f"{best_any['pct_candidates_in_AD']:.0f}%, but that is below "
                            f"--min-trainable-n={args.min_trainable_n})")
                logger.warning(
                    "[%s] no trainable subset size puts any candidate inside the AD%s. "
                    "Falling back to --min-n=%d.", receptor, note, args.min_n)

    # ---- selection ----
    passed = ann["terpene_score"] >= args.score_cutoff
    n_passed = int(passed.sum())
    used_min_n = False
    if n_passed < args.min_n:
        used_min_n = True
        n_take = min(args.min_n, len(ann))
        cutoff_idx = ann["terpene_score"].nlargest(n_take).index
        selected = ann.index.isin(cutoff_idx)
        logger.warning(
            "[%s] only %d molecules passed score >= %.2f; falling back to the "
            "top %d by terpene_score (effective cutoff %.3f)",
            receptor, n_passed, args.score_cutoff, n_take,
            float(ann.loc[cutoff_idx, "terpene_score"].min()),
        )
    else:
        selected = passed.values
    ann["selected"] = selected
    effective_cutoff = float(ann.loc[selected, "terpene_score"].min()) if selected.any() else np.nan

    # ---- AD coverage: full vs subset ----
    cov_full = cov_sub = None
    if len(cand_fps):
        cov_full = ad_coverage_report(fps, cand_fps, args.ad10_k, args.ad10_percentile, "full")
        cov_sub = ad_coverage_report(fps[selected], cand_fps, args.ad10_k,
                                     args.ad10_percentile, "subset")

    # ---- write outputs ----
    ann_path = os.path.join(receptor_dir, "06_terpene_annotated.csv")
    ann.to_csv(ann_path, index=False)

    subset_df = df.loc[df.index[keep_idx]][valid].reset_index(drop=True)[selected]
    sub_path = os.path.join(receptor_dir, "06_terpene_subset.csv")
    subset_df.to_csv(sub_path, index=False)

    def _clean_cov(c):
        if c is None:
            return None
        return {k: v for k, v in c.items() if not k.startswith("_")}

    rule_counts = {
        "obeys_C5_rule": int(ann["obeys_C5_rule"].sum()),
        "terpene_skeleton_size": int(ann["terpene_class"].ne("non-terpene-skeleton").sum()),
        "sesquiterpene_size": int(ann["is_sesquiterpene_size"].sum()),
        "is_terpenoid_strict": int(ann["is_terpenoid_strict"].sum()),
        "NP_score_gt_0": int((ann["NP_score"] > 0).sum()),
        "NP_score_gt_1": int((ann["NP_score"] > 1).sum()),
        "no_aromatic_rings": int((ann["n_aromatic_rings"] == 0).sum()),
        "no_nitrogen": int((ann["n_nitrogens"] == 0).sum()),
        "Fsp3_ge_0.6": int((ann["Fsp3"] >= 0.6).sum()),
        f"score_ge_{args.score_cutoff}": n_passed,
    }

    report = {
        "receptor": receptor,
        "source_file": src,
        "n_input_rows": int(len(df)),
        "n_parsable": int(len(ann)),
        "n_selected": int(selected.sum()),
        "score_cutoff_requested": args.score_cutoff,
        "score_cutoff_effective": effective_cutoff,
        "min_n": args.min_n,
        "min_n_fallback_used": used_min_n,
        "weights": WEIGHTS,
        "similarity_anchor_used": has_sim,
        "np_score_model": "RDKit NP_Score contrib (Ertl)" if np_scorer.available
                          else "surrogate fallback (NP_Score contrib unavailable)",
        "rule_counts": rule_counts,
        "np_score_full_mean": float(ann["NP_score"].mean()),
        "np_score_selected_mean": float(ann.loc[selected, "NP_score"].mean()) if selected.any() else None,
        "Fsp3_full_mean": float(ann["Fsp3"].mean()),
        "Fsp3_selected_mean": float(ann.loc[selected, "Fsp3"].mean()) if selected.any() else None,
        "ad_coverage_full": _clean_cov(cov_full),
        "ad_coverage_subset": _clean_cov(cov_sub),
        "subset_size_sweep": sweep_rows,
        "outputs": {"annotated": ann_path, "subset": sub_path},
    }
    with open(os.path.join(receptor_dir, "06_terpene_filter.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    # ---- plots ----
    if not args.no_plots:
        plots_dir = os.path.join(receptor_dir, "plots")
        cand_np = None
        if cand_smiles:
            cand_np = [np_scorer.score(m) for m in
                       (Chem.MolFromSmiles(s) for s in cand_smiles) if m is not None]
        try:
            plot_receptor(receptor, ann, pd.Series(selected, index=ann.index),
                          cov_full, cov_sub, cand_np, plots_dir)
            plot_chemical_space(receptor, fps, selected, cand_fps, plots_dir)
        except Exception as exc:
            logger.warning("[%s] plotting failed: %s", receptor, exc)

    msg = (f"[{receptor}] kept {int(selected.sum())}/{len(ann)} molecules "
           f"| NP {ann['NP_score'].mean():+.2f} -> "
           f"{ann.loc[selected, 'NP_score'].mean():+.2f}")
    if cov_full and cov_sub:
        msg += (f" | candidates in AD: {cov_full['pct_candidates_in_AD']:.0f}% -> "
                f"{cov_sub['pct_candidates_in_AD']:.0f}%")
    logger.info(msg)

    if cov_sub and cov_sub["n_candidates_in_AD"] == 0:
        gap = cov_sub["ad10_threshold"] - cov_sub["candidate_ad10_max"]
        logger.warning(
            "[%s] NO candidate falls inside the AD even after reduction. "
            "Best candidate AD10 = %.3f vs cutoff %.3f (gap %.3f). "
            "This dataset simply contains no chemistry close to the terpenes: "
            "%d/%d molecules are strict terpenoids, %d have no aromatic ring. "
            "Reducing it further cannot fix this — the training data itself "
            "needs terpene/natural-product bioactivity added.",
            receptor, cov_sub["candidate_ad10_max"], cov_sub["ad10_threshold"], gap,
            rule_counts["is_terpenoid_strict"], len(ann), rule_counts["no_aromatic_rings"],
        )

    return {
        "receptor": receptor,
        "source_file": src,
        "n_molecules": int(len(ann)),
        "n_selected": int(selected.sum()),
        "pct_selected": float(100.0 * selected.sum() / len(ann)),
        "score_cutoff_effective": effective_cutoff,
        "min_n_fallback_used": used_min_n,
        "n_terpenoid_strict": rule_counts["is_terpenoid_strict"],
        "n_NP_gt_0": rule_counts["NP_score_gt_0"],
        "n_sesquiterpene_size": rule_counts["sesquiterpene_size"],
        "mean_NP_score_full": float(ann["NP_score"].mean()),
        "mean_NP_score_selected": float(ann.loc[selected, "NP_score"].mean()) if selected.any() else np.nan,
        "mean_Fsp3_full": float(ann["Fsp3"].mean()),
        "mean_Fsp3_selected": float(ann.loc[selected, "Fsp3"].mean()) if selected.any() else np.nan,
        "ad10_threshold_full": cov_full["ad10_threshold"] if cov_full else np.nan,
        "ad10_threshold_subset": cov_sub["ad10_threshold"] if cov_sub else np.nan,
        "n_candidates_in_AD_full": cov_full["n_candidates_in_AD"] if cov_full else np.nan,
        "n_candidates_in_AD_subset": cov_sub["n_candidates_in_AD"] if cov_sub else np.nan,
        "pct_candidates_in_AD_full": cov_full["pct_candidates_in_AD"] if cov_full else np.nan,
        "pct_candidates_in_AD_subset": cov_sub["pct_candidates_in_AD"] if cov_sub else np.nan,
        "candidate_ad10_max_subset": cov_sub["candidate_ad10_max"] if cov_sub else np.nan,
        "ad_gap_subset": (cov_sub["ad10_threshold"] - cov_sub["candidate_ad10_max"]) if cov_sub else np.nan,
        "verdict": (
            "no candidate reachable — dataset lacks terpene chemistry"
            if cov_sub and cov_sub["n_candidates_in_AD"] == 0 else
            "partial AD coverage achieved"
            if cov_sub and cov_sub["pct_candidates_in_AD"] < 50 else
            "good AD coverage achieved" if cov_sub else "no candidates supplied"
        ),
    }


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Enrich QSAR datasets for terpenes / terpenoids / isoprenoids / "
                    "natural-product-like molecules so that terpene predictions fall "
                    "inside the applicability domain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--receptor", help="single receptor folder name under output/")
    g.add_argument("--all", action="store_true", help="process every receptor under output/")

    ap.add_argument("--output-dir", default=None,
                    help="base output directory (default: config.BASE_OUTPUT_DIR or ./output)")
    ap.add_argument("--candidates", default=None,
                    help="CSV of the molecules you want to predict (SMILES column). "
                         "Enables the similarity sub-score and the AD coverage report.")
    ap.add_argument("--score-cutoff", type=float, default=DEFAULT_SCORE_CUTOFF,
                    help="minimum terpene_score to keep a molecule")
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                    help="if fewer molecules pass the cutoff, keep the top-N by score instead")
    ap.add_argument("--morgan-radius", type=int, default=DEFAULT_MORGAN_RADIUS)
    ap.add_argument("--morgan-nbits", type=int, default=DEFAULT_MORGAN_NBITS)
    ap.add_argument("--ad10-k", type=int, default=DEFAULT_AD10_K)
    ap.add_argument("--ad10-percentile", type=float, default=DEFAULT_AD10_PERCENTILE)
    ap.add_argument("--optimize-n", action="store_true",
                    help="sweep several subset sizes and keep the one that puts the most "
                         "candidates inside the AD (requires --candidates). The AD10 cutoff "
                         "is the 5th percentile of the training set's own similarity "
                         "distribution, so a smaller/tighter subset lowers the bar.")
    ap.add_argument("--sweep-sizes", type=int, nargs="+",
                    default=[50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000],
                    help="subset sizes tried by --optimize-n")
    ap.add_argument("--min-trainable-n", type=int, default=100,
                    help="never let --optimize-n pick a subset smaller than this")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--emit-retrain-cmd", action="store_true",
                    help="print the commands to retrain on the reduced datasets")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    base = args.output_dir or (
        getattr(pipeline_config, "BASE_OUTPUT_DIR", None)
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    )
    if not os.path.isdir(base):
        logger.error("Output directory not found: %s", base)
        sys.exit(1)

    receptors = (
        sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        if args.all else [args.receptor]
    )
    logger.info("Processing %d receptor(s) under %s", len(receptors), base)

    np_scorer = NPScorer()

    cand_smiles, cand_names = [], []
    if args.candidates:
        cand_smiles, cand_names = load_candidates(args.candidates)
        logger.info("Loaded %d candidate molecules from %s", len(cand_smiles), args.candidates)

    rows = []
    for rec in receptors:
        try:
            r = process_receptor(rec, base, np_scorer, cand_smiles, cand_names, args)
            if r:
                rows.append(r)
        except Exception as exc:
            logger.exception("[%s] failed: %s", rec, exc)

    if not rows:
        logger.error("No receptor produced results.")
        sys.exit(1)

    summary = pd.DataFrame(rows)
    summary_csv = os.path.join(base, "06_terpene_summary.csv")
    summary.to_csv(summary_csv, index=False)
    with open(os.path.join(base, "06_terpene_summary.json"), "w") as fh:
        json.dump({
            "n_receptors": len(rows),
            "candidates_file": args.candidates,
            "n_candidates": len(cand_smiles),
            "score_cutoff": args.score_cutoff,
            "min_n": args.min_n,
            "weights": WEIGHTS,
            "np_score_model": "RDKit NP_Score contrib (Ertl)" if np_scorer.available
                              else "surrogate fallback",
            "receptors": rows,
        }, fh, indent=2, default=float)

    if not args.no_plots and len(rows) > 1:
        try:
            plot_summary(summary, os.path.join(base, "06_terpene_summary_plots.png"))
        except Exception as exc:
            logger.warning("Summary plot failed: %s", exc)

    # ---- console summary ----
    cols = ["receptor", "n_molecules", "n_selected", "n_terpenoid_strict",
            "mean_NP_score_full", "mean_NP_score_selected",
            "pct_candidates_in_AD_full", "pct_candidates_in_AD_subset",
            "ad_gap_subset", "verdict"]
    cols = [c for c in cols if c in summary.columns]
    print("\n" + "=" * 100)
    print("TERPENE / NATURAL-PRODUCT ENRICHMENT SUMMARY")
    print("=" * 100)
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=" * 100)
    print(f"Summary written to {summary_csv}")

    if args.emit_retrain_cmd:
        print("\nTo retrain on the reduced datasets:")
        for r in rows:
            print(f'  python step4_rf_model.py --receptor "{r["receptor"]}" '
                  f'--input_csv "{os.path.join(base, r["receptor"], "06_terpene_subset.csv")}"')


if __name__ == "__main__":
    main()
