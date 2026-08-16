"""
ad_uncertainty.py — Shared applicability domain (Tanimoto AD10) and
uncertainty quantification (RF tree variance + Mondrian normalized
conformal prediction) utilities for the QSAR pipeline.

Used by step4_rf_model.py, step5_chemprop_model.py, and predict.py so
that AD/CP logic and artifact formats stay consistent across models.

Key ideas
---------
1. AD10 (Tanimoto kNN applicability domain):
   For a query fingerprint, AD10 = mean Tanimoto similarity to its
   k=10 nearest neighbors in a reference (training) fingerprint set.
   A molecule is "in domain" if AD10 >= threshold, where threshold is
   a low percentile (default 5th) of the *training set's own*
   leave-one-out AD10 distribution.

2. RF uncertainty (tree-to-tree variance):
   - For calibration, we use *out-of-bag* (OOB) predictions: for each
     training sample, only trees that did not see it in their
     bootstrap sample vote. This gives an honest "as if unseen"
     prediction/uncertainty pair without sacrificing training data
     for a dedicated calibration split.
   - For test / new molecules, the *full* ensemble (all trees) is
     used to compute the mean prediction and std.

3. Mondrian normalized conformal prediction (RF):
   Nonconformity score  a_i = |y_i - yhat_i| / (sigma_i + beta)
   computed on OOB calibration data, split into two strata ("in AD"
   / "out of AD"). Each stratum gets its own conformal quantile q,
   so that interval width adapts to local model disagreement AND
   coverage is honest conditional on AD status, not just marginal.
   Prediction interval for a new point: yhat +/- q_stratum * (sigma + beta)

4. Mondrian conformal prediction (Chemprop):
   Chemprop has no internal ensemble variance, so nonconformity score
   is simply |y - yhat| computed on the validation set (used here as
   the calibration set — note this reuses data involved in checkpoint
   selection, a mild, commonly-accepted source of optimism in
   coverage). Same Mondrian (in-AD / out-of-AD) stratification is
   applied. Prediction interval: yhat +/- q_stratum
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smiles_list_to_fp_matrix(smiles_list, radius: int, n_bits: int):
    """
    Morgan fingerprints for a list of SMILES (used for AD purposes by
    models, like Chemprop, that don't natively operate on fingerprints).
    Returns (fp_matrix, valid_mask) — invalid SMILES get an all-zero row
    and valid_mask=False at that position, so caller can align/filter.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.int8)
    valid_mask = np.zeros(len(smiles_list), dtype=bool)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr
        valid_mask[i] = True

    return fps, valid_mask


# ─────────────────────────────────────────────────────────────
# Tanimoto similarity / AD10
# ─────────────────────────────────────────────────────────────
def tanimoto_similarity_matrix(query_fps: np.ndarray, reference_fps: np.ndarray) -> np.ndarray:
    """
    Vectorized Tanimoto similarity between every row of *query_fps*
    (m x nbits binary) and every row of *reference_fps* (n x nbits binary).
    Returns an (m x n) similarity matrix.
    """
    query_fps = np.asarray(query_fps, dtype=np.float64)
    reference_fps = np.asarray(reference_fps, dtype=np.float64)

    dot = query_fps @ reference_fps.T                      # (m, n) intersection counts
    sum_q = query_fps.sum(axis=1, keepdims=True)            # (m, 1)
    sum_r = reference_fps.sum(axis=1, keepdims=True).T      # (1, n)
    union = sum_q + sum_r - dot

    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, dot / union, 0.0)
    return sim


def compute_ad10(
    query_fps: np.ndarray,
    reference_fps: np.ndarray,
    k: int = 10,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Mean Tanimoto similarity to the k nearest neighbors in reference_fps,
    for each row in query_fps.

    If exclude_self=True, assumes query_fps IS reference_fps (leave-one-out
    computation for building the training reference distribution): the
    best self-match (similarity == itself) at each row is masked out
    before taking the top-k.
    """
    sim = tanimoto_similarity_matrix(query_fps, reference_fps)
    n_ref = reference_fps.shape[0]
    k_eff = min(k, n_ref - 1 if exclude_self else n_ref)
    if k_eff <= 0:
        raise ValueError("Not enough reference molecules to compute AD10.")

    if exclude_self:
        # zero out the diagonal (self-similarity) row-by-row
        idx = np.arange(sim.shape[0])
        sim = sim.copy()
        sim[idx, idx] = -1.0  # guarantees it's excluded from top-k

    # top-k similarities per row (partition is O(n) vs full sort)
    part = np.partition(sim, -k_eff, axis=1)[:, -k_eff:]
    ad10 = part.mean(axis=1)
    return ad10


def ad_threshold(train_ad10_scores: np.ndarray, percentile: float = 5.0) -> float:
    """Low-percentile cutoff of the training set's own AD10 distribution."""
    return float(np.percentile(train_ad10_scores, percentile))


# ─────────────────────────────────────────────────────────────
# Random Forest ensemble / OOB utilities
# ─────────────────────────────────────────────────────────────
def rf_ensemble_mean_std(rf_model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-ensemble (all trees) mean prediction and std for arbitrary X."""
    tree_preds = np.stack([tree.predict(X) for tree in rf_model.estimators_], axis=0)
    return tree_preds.mean(axis=0), tree_preds.std(axis=0)


def rf_oob_mean_std(rf_model, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each training sample, average/std across only the trees for which
    that sample was out-of-bag (not in that tree's bootstrap sample).

    Returns (oob_mean, oob_std, valid_mask) where valid_mask marks samples
    that had at least one OOB tree (essentially always true for
    n_estimators=100, but guarded against edge cases).
    """
    n_samples = X_train.shape[0]
    n_trees = len(rf_model.estimators_)
    oob_pred_matrix = np.full((n_samples, n_trees), np.nan, dtype=np.float64)

    for t, (tree, sample_indices) in enumerate(
        zip(rf_model.estimators_, rf_model.estimators_samples_)
    ):
        in_bag = np.zeros(n_samples, dtype=bool)
        in_bag[sample_indices] = True
        oob_idx = np.where(~in_bag)[0]
        if len(oob_idx) == 0:
            continue
        oob_pred_matrix[oob_idx, t] = tree.predict(X_train[oob_idx])

    valid_mask = ~np.all(np.isnan(oob_pred_matrix), axis=1)
    oob_mean = np.full(n_samples, np.nan)
    oob_std = np.full(n_samples, np.nan)
    oob_mean[valid_mask] = np.nanmean(oob_pred_matrix[valid_mask], axis=1)
    oob_std[valid_mask] = np.nanstd(oob_pred_matrix[valid_mask], axis=1)
    return oob_mean, oob_std, valid_mask


# ─────────────────────────────────────────────────────────────
# Mondrian (AD-stratified) conformal prediction
# ─────────────────────────────────────────────────────────────
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Finite-sample-corrected conformal quantile of nonconformity scores,
    for a (1 - alpha) coverage target.
    """
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute a conformal quantile from zero calibration points.")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def mondrian_calibrate(
    nonconformity_scores: np.ndarray,
    in_ad_mask: np.ndarray,
    alpha: float,
    min_stratum_size: int = 20,
) -> dict:
    """
    Split calibration nonconformity scores by AD status and compute a
    separate conformal quantile per stratum. Falls back to a single
    pooled quantile for a stratum that's too small to calibrate reliably.
    """
    nonconformity_scores = np.asarray(nonconformity_scores, dtype=np.float64)
    in_ad_mask = np.asarray(in_ad_mask, dtype=bool)

    scores_in = nonconformity_scores[in_ad_mask]
    scores_out = nonconformity_scores[~in_ad_mask]
    pooled_q = conformal_quantile(nonconformity_scores, alpha)

    result = {
        "alpha": alpha,
        "n_in": int(len(scores_in)),
        "n_out": int(len(scores_out)),
        "pooled_q": pooled_q,
        "fallback_used": {"in": False, "out": False},
    }

    if len(scores_in) >= min_stratum_size:
        result["q_in"] = conformal_quantile(scores_in, alpha)
    else:
        result["q_in"] = pooled_q
        result["fallback_used"]["in"] = True

    if len(scores_out) >= min_stratum_size:
        result["q_out"] = conformal_quantile(scores_out, alpha)
    else:
        result["q_out"] = pooled_q
        result["fallback_used"]["out"] = True

    return result


def mondrian_quantile_for(cal: dict, in_ad: bool) -> float:
    return cal["q_in"] if in_ad else cal["q_out"]


def conformal_interval(
    pred: np.ndarray,
    q: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    beta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build prediction intervals.
    - Normalized (RF): half_width = q * (sigma + beta)
    - Plain (Chemprop): pass sigma=None -> half_width = q
    """
    pred = np.asarray(pred, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        half_width = q * (sigma + beta)
    else:
        half_width = q * np.ones_like(pred)
    return pred - half_width, pred + half_width, half_width


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def coverage_sweep(
    nonconformity_scores: np.ndarray,
    in_ad_mask_cal: np.ndarray,
    y_true_test: np.ndarray,
    pred_test: np.ndarray,
    in_ad_mask_test: np.ndarray,
    sigma_test: Optional[np.ndarray] = None,
    beta: float = 0.0,
    alphas: Optional[list[float]] = None,
    min_stratum_size: int = 20,
) -> list[dict]:
    """Empirical coverage on the test set across a sweep of nominal alphas."""
    if alphas is None:
        alphas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

    rows = []
    for alpha in alphas:
        cal = mondrian_calibrate(nonconformity_scores, in_ad_mask_cal, alpha, min_stratum_size)
        q_test = np.where(in_ad_mask_test, cal["q_in"], cal["q_out"])
        lower, upper, width = conformal_interval(pred_test, q_test, sigma_test, beta)
        cov = empirical_coverage(y_true_test, lower, upper)
        rows.append({
            "alpha": alpha,
            "nominal_coverage": 1 - alpha,
            "empirical_coverage": cov,
            "mean_interval_width": float(np.mean(width)),
        })
    return rows


# ─────────────────────────────────────────────────────────────
# Binning helper
# ─────────────────────────────────────────────────────────────
def bin_average(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Equal-count (quantile) binning of x, returning bin-center x and mean y per bin."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    edges = np.array_split(np.arange(len(x_sorted)), n_bins)
    bin_x, bin_y, bin_n = [], [], []
    for idx in edges:
        if len(idx) == 0:
            continue
        bin_x.append(x_sorted[idx].mean())
        bin_y.append(y_sorted[idx].mean())
        bin_n.append(len(idx))
    return np.array(bin_x), np.array(bin_y), np.array(bin_n)


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────
def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {path}")


def plot_ad_distribution(train_ad10, test_ad10, threshold, out_path, model_name):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_ad10, bins=40, alpha=0.55, label="Training (leave-one-out)", color="steelblue")
    ax.hist(test_ad10, bins=40, alpha=0.55, label="Test", color="darkorange")
    ax.axvline(threshold, color="red", linestyle="--", lw=1.5,
                label=f"AD10 threshold = {threshold:.3f}")
    ax.set_xlabel("AD10 (mean Tanimoto similarity to 10 nearest training neighbors)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{model_name}: Applicability Domain (Tanimoto AD10)", fontsize=13)
    ax.legend()
    _save(fig, out_path)


def plot_ad_williams(ad10, std_residuals, threshold, out_path, model_name):
    """Williams-plot analogue: AD10 similarity (x) vs standardized residual (y)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    in_ad = ad10 >= threshold
    ax.scatter(ad10[in_ad], std_residuals[in_ad], s=14, alpha=0.55, color="steelblue", label="In AD")
    ax.scatter(ad10[~in_ad], std_residuals[~in_ad], s=14, alpha=0.7, color="crimson", label="Out of AD")
    ax.axvline(threshold, color="red", linestyle="--", lw=1.2)
    ax.axhline(3, color="gray", linestyle=":", lw=1)
    ax.axhline(-3, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("AD10 similarity", fontsize=11)
    ax.set_ylabel("Standardized residual", fontsize=11)
    ax.set_title(f"{model_name}: AD10-Williams plot (test set)", fontsize=13)
    ax.legend()
    _save(fig, out_path)


def plot_pred_vs_actual_ci(
    y_train, pred_train, y_test, pred_test, lower_test, upper_test,
    target_col, out_path, model_name,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    ax = axes[0]
    ax.scatter(y_train, pred_train, s=10, alpha=0.35, color="steelblue")
    lo = min(y_train.min(), pred_train.min())
    hi = max(y_train.max(), pred_train.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.3)
    ax.set_xlabel(f"Actual {target_col}")
    ax.set_ylabel(f"Predicted {target_col}")
    ax.set_title("Training set (fit quality only — no CP guarantee applies here)", fontsize=11)

    ax = axes[1]
    err = np.abs(upper_test - lower_test) / 2
    order = np.argsort(y_test)
    ax.errorbar(
        y_test[order], pred_test[order], yerr=err[order],
        fmt="o", ms=3.5, alpha=0.5, ecolor="gray", elinewidth=0.7, capsize=0,
        color="darkorange",
    )
    lo = min(y_test.min(), lower_test.min())
    hi = max(y_test.max(), upper_test.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.3)
    ax.set_xlabel(f"Actual {target_col}")
    ax.set_ylabel(f"Predicted {target_col}")
    ax.set_title("Test set with Mondrian-calibrated intervals", fontsize=11)

    fig.suptitle(f"{model_name}: Predicted vs Actual {target_col}", fontsize=14)
    fig.tight_layout()
    _save(fig, out_path)


def plot_metric_vs_error_binned(metric, abs_error, metric_name, out_path, model_name, n_bins=10):
    bin_x, bin_y, bin_n = bin_average(metric, abs_error, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(metric, abs_error, s=10, alpha=0.25, color="gray", label="Test molecules")
    ax.plot(bin_x, bin_y, "o-", color="crimson", lw=2, ms=6, label=f"Bin-averaged trend (n={n_bins})")
    ax.set_xlabel(metric_name, fontsize=11)
    ax.set_ylabel("Absolute error", fontsize=11)
    ax.set_title(f"{model_name}: {metric_name} vs Absolute Error (test set)", fontsize=13)
    ax.legend()
    _save(fig, out_path)


def plot_coverage_calibration(coverage_rows, out_path, model_name):
    nominal = [r["nominal_coverage"] for r in coverage_rows]
    empirical = [r["empirical_coverage"] for r in coverage_rows]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal calibration")
    ax.plot(nominal, empirical, "o-", color="steelblue", lw=2)
    ax.set_xlabel("Nominal coverage (1 - alpha)", fontsize=11)
    ax.set_ylabel("Empirical coverage (test set)", fontsize=11)
    ax.set_title(f"{model_name}: Conformal Coverage Calibration", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    _save(fig, out_path)


def plot_ad_reliability_heatmap(in_ad_mask, abs_error, error_threshold, out_path, model_name):
    reliable = abs_error <= error_threshold
    counts = np.array([
        [np.sum(in_ad_mask & reliable), np.sum(in_ad_mask & ~reliable)],
        [np.sum(~in_ad_mask & reliable), np.sum(~in_ad_mask & ~reliable)],
    ])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(counts, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=13,
                     color="white" if counts[i, j] > counts.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Error ≤ {error_threshold:.2f}", f"Error > {error_threshold:.2f}"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["In AD", "Out of AD"])
    ax.set_title(f"{model_name}: AD status vs Reliability (test set)", fontsize=12)
    fig.colorbar(im, ax=ax, label="Count")
    _save(fig, out_path)


def plot_width_vs_metric(width, metric, metric_name, out_path, model_name):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(metric, width, s=10, alpha=0.35, color="seagreen")
    ax.set_xlabel(metric_name, fontsize=11)
    ax.set_ylabel("Conformal interval width", fontsize=11)
    ax.set_title(f"{model_name}: Interval Width vs {metric_name} (test set)", fontsize=13)
    _save(fig, out_path)


def plot_new_molecules_ad10(train_ad10, new_ad10, threshold, out_path, model_name, names=None):
    """Overlay new molecules on the training AD10 reference distribution."""
    new_ad10 = np.asarray(new_ad10, dtype=np.float64)
    in_ad = new_ad10 >= threshold

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(train_ad10, bins=40, alpha=0.5, color="steelblue", label="Training (reference)")
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"AD10 threshold = {threshold:.3f}")

    ymax = ax.get_ylim()[1]
    rng = np.random.default_rng(0)
    jitter = rng.uniform(0.03 * ymax, 0.18 * ymax, size=len(new_ad10))
    ax.scatter(new_ad10[in_ad], jitter[in_ad], color="seagreen", s=55, marker="o",
                edgecolor="black", linewidth=0.6, label="New molecules (in AD)", zorder=5)
    ax.scatter(new_ad10[~in_ad], jitter[~in_ad], color="crimson", s=55, marker="X",
                edgecolor="black", linewidth=0.6, label="New molecules (out of AD)", zorder=5)

    if names is not None and len(names) <= 25:
        for x, y, n in zip(new_ad10, jitter, names):
            ax.annotate(str(n), (x, y), fontsize=7, xytext=(0, 6),
                         textcoords="offset points", ha="center")

    ax.set_xlabel("AD10 (mean Tanimoto similarity to 10 nearest training neighbors)", fontsize=11)
    ax.set_ylabel("Training count", fontsize=11)
    ax.set_title(f"{model_name}: New Molecules vs Training AD10 Distribution", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, out_path)


def plot_ad10_ranked(new_ad10, names, threshold, out_path, model_name):
    """Per-molecule AD10 ranked bar chart — how far each candidate sits from the AD boundary."""
    new_ad10 = np.asarray(new_ad10, dtype=np.float64)
    names = [str(n) for n in names]
    order = np.argsort(new_ad10)
    ad_sorted = new_ad10[order]
    names_sorted = [names[i] for i in order]
    colors = ["seagreen" if v >= threshold else "crimson" for v in ad_sorted]

    show_labels = len(ad_sorted) <= 60
    fig_height = max(4.0, 0.28 * len(ad_sorted) + 1.5) if show_labels else 6.0
    fig, ax = plt.subplots(figsize=(9, min(fig_height, 30)))
    y_pos = np.arange(len(ad_sorted))
    ax.barh(y_pos, ad_sorted, color=colors, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axvline(threshold, color="black", linestyle="--", lw=1.3, label=f"Threshold = {threshold:.3f}")

    if show_labels:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_sorted, fontsize=8 if len(ad_sorted) <= 30 else 6)
    else:
        ax.set_ylabel("Molecule rank (labels omitted — too many candidates)", fontsize=10)
        ax.set_yticks([])

    ax.set_xlabel("AD10 similarity", fontsize=11)
    ax.set_title(f"{model_name}: New Molecules Ranked by Distance from AD Boundary", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out_path)


def plot_chemical_space_pca(
    train_fps, new_fps, new_in_ad_mask, out_path, model_name,
    names=None, max_train_points=3000,
):
    """
    2D PCA of Morgan fingerprints: training cloud vs. where new molecules land.
    Caveat (worth remembering when reading this): 2 PCA components typically capture
    only a small fraction of variance in sparse, high-dimensional binary fingerprints —
    this is a qualitative, descriptive picture of chemical space, not the metric that
    actually drives the AD10/CP decision. Use it alongside the AD10 plots above, not
    instead of them.
    """
    from sklearn.decomposition import PCA

    train_fps = np.asarray(train_fps, dtype=np.float64)
    new_fps = np.asarray(new_fps, dtype=np.float64)
    new_in_ad_mask = np.asarray(new_in_ad_mask, dtype=bool)

    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    pca.fit(train_fps)

    rng = np.random.default_rng(0)
    if train_fps.shape[0] > max_train_points:
        idx = rng.choice(train_fps.shape[0], size=max_train_points, replace=False)
        train_plot_2d = pca.transform(train_fps[idx])
    else:
        train_plot_2d = pca.transform(train_fps)

    new_2d = pca.transform(new_fps)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(train_plot_2d[:, 0], train_plot_2d[:, 1], s=8, alpha=0.25,
                color="gray", label="Training set")
    ax.scatter(new_2d[new_in_ad_mask, 0], new_2d[new_in_ad_mask, 1], s=60, color="seagreen",
                edgecolor="black", linewidth=0.6, marker="o", label="New molecules (in AD)", zorder=5)
    ax.scatter(new_2d[~new_in_ad_mask, 0], new_2d[~new_in_ad_mask, 1], s=60, color="crimson",
                edgecolor="black", linewidth=0.6, marker="X", label="New molecules (out of AD)", zorder=5)

    if names is not None and len(names) <= 25:
        for (x, y), n in zip(new_2d, names):
            ax.annotate(str(n), (x, y), fontsize=7, xytext=(0, 6),
                         textcoords="offset points", ha="center")

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title(f"{model_name}: Chemical Space (PCA of Morgan Fingerprints)", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# Save / load artifacts
# ─────────────────────────────────────────────────────────────
def save_ad_uncertainty_artifacts(
    out_dir: str,
    prefix: str,
    train_fps: np.ndarray,
    ad_k: int,
    ad_threshold_value: float,
    calibration: dict,
    beta: float,
    train_ad10: Optional[np.ndarray] = None,
    extra_json: Optional[dict] = None,
) -> tuple[str, str]:
    """
    Saves:
      {prefix}_ad_train_fps.npz   -> training fingerprint matrix (for kNN lookups on
                                      new molecules) + the training set's own LOO AD10
                                      distribution (for reference plots)
      {prefix}_ad_uncertainty.json -> AD threshold, Mondrian quantiles, beta, config
    Returns (npz_path, json_path).
    """
    npz_path = os.path.join(out_dir, f"{prefix}_ad_train_fps.npz")
    save_kwargs = {"train_fps": train_fps.astype(np.int8)}
    if train_ad10 is not None:
        save_kwargs["train_ad10"] = np.asarray(train_ad10, dtype=np.float64)
    np.savez_compressed(npz_path, **save_kwargs)

    payload = {
        "ad_k": ad_k,
        "ad_threshold": ad_threshold_value,
        "beta": beta,
        "calibration": calibration,
        "train_fps_path": npz_path,
    }
    if extra_json:
        payload.update(extra_json)

    json_path = os.path.join(out_dir, f"{prefix}_ad_uncertainty.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return npz_path, json_path


def load_ad_uncertainty_artifacts(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    data = np.load(payload["train_fps_path"])
    payload["train_fps"] = data["train_fps"]
    payload["train_ad10"] = data["train_ad10"] if "train_ad10" in data else None
    return payload
