"""
Step 4 — Train a Random Forest regressor on Morgan fingerprints.

Pipeline:
  1. Generate Morgan fingerprints (radius=2, nBits=2048)
  2. 80/20 train/test split
  3. Train RandomForestRegressor with default hyperparameters
  4. Evaluate (R², RMSE, MAE) and generate plots
  5. Serialize model + config with joblib

Usage:
    python step4_rf_model.py --receptor FGFR1
"""

import argparse
import json
import os
import warnings
#import shap #Also moving this one to the main pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
import config
import ad_uncertainty as adu

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Fingerprint Generation ────────────────────────────────────
def smiles_to_morgan_fp(smiles: str, generator):
    """Convert a SMILES string to a Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def generate_fingerprints(
    smiles_list: list[str],
    radius: int = config.MORGAN_RADIUS,
    n_bits: int = config.MORGAN_NBITS,
) -> tuple[np.ndarray, list[int]]:
    """
    Generate Morgan FPs for a list of SMILES.
    Returns (feature_matrix, valid_indices).
    """
    fps = []
    valid_idx = []

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan_fp(smi, generator)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    return np.array(fps), valid_idx


# ── Main ──────────────────────────────────────────────────────
def run(receptor_name: str, bioactivity_type: str | None = None) -> str:
    """
    Main entry point for Step 4.
    Returns the path to the saved model.
    """
    bio_type = bioactivity_type or config.BIOACTIVITY_TYPE
    out_dir = config.get_output_dir(receptor_name)
    models_dir = config.get_models_dir(receptor_name)
    plots_dir = config.get_plots_dir(receptor_name)
    preproc_path = os.path.join(out_dir, "02_preprocessed.csv")

    print(f"[Step 4] Random Forest model for '{receptor_name}'")

    # ── Load data ─────────────────────────────────────────────
    df = pd.read_csv(preproc_path)
    convert = config.CONVERT_TO_PIC50 and bio_type.upper() in ("IC50", "KI")
    target_col = "pIC50" if convert else "standard_value"
    print(f"  Loaded {len(df)} rows, target column: {target_col}")

    # ── Generate fingerprints ─────────────────────────────────
    print(f"  Generating Morgan FPs (radius={config.MORGAN_RADIUS}, "
          f"nBits={config.MORGAN_NBITS})...")
    X, valid_idx = generate_fingerprints(df["Smiles"].tolist())
    y = df[target_col].values[valid_idx]
    print(f"  Valid molecules: {len(valid_idx)}/{len(df)}")

    # ── Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.RF_TEST_SIZE,
        random_state=config.RF_RANDOM_STATE,
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Train ─────────────────────────────────────────────────
    print("  Training Random Forest (default hyperparameters)...")
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=config.RF_RANDOM_STATE,
        n_jobs=-1,
    )

    # Perform cross-validation
    # 5-fold cross-validation on the training set
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RF_RANDOM_STATE,
    )

    cv_scores = cross_val_score(
        rf,
        X_train,
        y_train,
        cv=kf,
        scoring="r2",
        n_jobs=-1,
    )

    cv_results = {
        "r2_scores": [float(score) for score in cv_scores],
        "mean_r2": float(np.mean(cv_scores)),
        "std_r2": float(np.std(cv_scores)),
    }

    cv_path = os.path.join(models_dir, "rf_cross_validation.json")
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"  CV results -> {cv_path}")
    print(f"    CV mean R2: {cv_results['mean_r2']:.4f}")
    print(f"    CV std R2: {cv_results['std_r2']:.4f}")

    rf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    metrics = {
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
    }
    print("  ── Metrics ──")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # ── Plots ─────────────────────────────────────────────────
    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred_test, alpha=0.4, s=10)
    lo = min(y_test.min(), y_pred_test.min())
    hi = max(y_test.max(), y_pred_test.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_xlabel(f"Actual {target_col}", fontsize=12)
    ax.set_ylabel(f"Predicted {target_col}", fontsize=12)
    ax.set_title(
        f"RF: Predicted vs Actual  (R²={metrics['test_r2']:.3f})", fontsize=14
    )
    path = os.path.join(plots_dir, "rf_pred_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {path}")

    # Residual distribution
    residuals = y_test - y_pred_test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Residual", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("RF: Residual Distribution", fontsize=14)
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    path = os.path.join(plots_dir, "rf_residuals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {path}")

    # ── Applicability Domain (Tanimoto AD10) ───────────────────
    print("  Computing applicability domain (Tanimoto AD10)...")
    train_ad10_loo = adu.compute_ad10(X_train, X_train, k=config.AD10_K, exclude_self=True)
    ad10_thresh = adu.ad_threshold(train_ad10_loo, percentile=config.AD10_PERCENTILE)
    test_ad10 = adu.compute_ad10(X_test, X_train, k=config.AD10_K, exclude_self=False)
    in_ad_test = test_ad10 >= ad10_thresh
    print(f"    AD10 threshold ({config.AD10_PERCENTILE}th pct of training LOO AD10): {ad10_thresh:.4f}")
    print(f"    Test set in-AD: {in_ad_test.sum()}/{len(in_ad_test)}")

    adu.plot_ad_distribution(
        train_ad10_loo, test_ad10, ad10_thresh,
        os.path.join(plots_dir, "rf_ad10_distribution.png"), "RF",
    )

    # ── RF uncertainty: OOB calibration + full-ensemble test/deploy std ──
    print("  Computing OOB predictions/std for conformal calibration...")
    oob_mean, oob_std, oob_valid = adu.rf_oob_mean_std(rf, X_train)
    beta = float(np.median(oob_std[oob_valid]))  # regularization to avoid divide-by-~0
    print(f"    OOB-valid training samples: {oob_valid.sum()}/{len(oob_valid)}  |  beta={beta:.4f}")

    cal_scores = np.abs(y_train[oob_valid] - oob_mean[oob_valid]) / (oob_std[oob_valid] + beta)
    in_ad_train_oob = (train_ad10_loo >= ad10_thresh)[oob_valid]

    cp_calibration = adu.mondrian_calibrate(
        cal_scores, in_ad_train_oob, alpha=config.CP_ALPHA,
        min_stratum_size=config.CP_MIN_STRATUM_SIZE,
    )
    print(f"    Mondrian conformal calibration (alpha={config.CP_ALPHA}): {cp_calibration}")

    # Full-ensemble mean/std for the test set (as would be done for new molecules)
    test_ensemble_mean, test_ensemble_std = adu.rf_ensemble_mean_std(rf, X_test)
    q_test = np.where(in_ad_test, cp_calibration["q_in"], cp_calibration["q_out"])
    lower_test, upper_test, width_test = adu.conformal_interval(
        test_ensemble_mean, q_test, sigma=test_ensemble_std, beta=beta
    )
    test_coverage = adu.empirical_coverage(y_test, lower_test, upper_test)
    print(f"    Empirical test coverage @ {1 - config.CP_ALPHA:.0%} target: {test_coverage:.3f}")

    coverage_rows = adu.coverage_sweep(
        cal_scores, in_ad_train_oob, y_test, test_ensemble_mean, in_ad_test,
        sigma_test=test_ensemble_std, beta=beta, alphas=config.CP_COVERAGE_SWEEP_ALPHAS,
        min_stratum_size=config.CP_MIN_STRATUM_SIZE,
    )

    # ── AD / uncertainty plots ──────────────────────────────────
    std_residuals = (y_test - y_pred_test) / (metrics["test_rmse"] + 1e-9)
    adu.plot_ad_williams(
        test_ad10, std_residuals, ad10_thresh,
        os.path.join(plots_dir, "rf_ad10_williams.png"), "RF",
    )
    adu.plot_pred_vs_actual_ci(
        y_train, y_pred_train, y_test, test_ensemble_mean, lower_test, upper_test,
        target_col, os.path.join(plots_dir, "rf_pred_vs_actual_ci.png"), "RF",
    )
    test_abs_error = np.abs(y_test - test_ensemble_mean)
    adu.plot_metric_vs_error_binned(
        test_ad10, test_abs_error, "AD10 similarity",
        os.path.join(plots_dir, "rf_ad10_vs_error.png"), "RF",
    )
    adu.plot_metric_vs_error_binned(
        test_ensemble_std, test_abs_error, "RF tree-to-tree std (uncertainty)",
        os.path.join(plots_dir, "rf_uncertainty_vs_error.png"), "RF",
    )
    adu.plot_coverage_calibration(
        coverage_rows, os.path.join(plots_dir, "rf_coverage_calibration.png"), "RF",
    )
    adu.plot_ad_reliability_heatmap(
        in_ad_test, test_abs_error, config.AD_RELIABILITY_ERROR_THRESHOLD,
        os.path.join(plots_dir, "rf_ad_reliability_heatmap.png"), "RF",
    )
    adu.plot_width_vs_metric(
        width_test, test_ensemble_std, "RF tree-to-tree std (uncertainty)",
        os.path.join(plots_dir, "rf_interval_width_vs_uncertainty.png"), "RF",
    )

    # ── Save AD / uncertainty artifacts for reuse in predict.py ─
    npz_path, adu_json_path = adu.save_ad_uncertainty_artifacts(
        out_dir=models_dir,
        prefix="rf",
        train_fps=X_train,
        ad_k=config.AD10_K,
        ad_threshold_value=ad10_thresh,
        calibration=cp_calibration,
        beta=beta,
        train_ad10=train_ad10_loo,
        extra_json={
            "target_col": target_col,
            "test_coverage_at_target_alpha": test_coverage,
            "coverage_sweep": coverage_rows,
            "morgan_radius": config.MORGAN_RADIUS,
            "morgan_nbits": config.MORGAN_NBITS,
        },
    )
    print(f"    AD/uncertainty artifacts -> {adu_json_path}  |  {npz_path}")

    # SHAP analysis
    print("  Running SHAP analysis...")
    print("Importing SHAP")
    import shap
    print("SHAP imported, performing analysis...")

    feature_names = [f"bit_{i}" for i in range(config.MORGAN_NBITS)]

    shap_sample_size = min(500, len(X_test))
    background_size = min(200, len(X_train))

    shap_rng = np.random.default_rng(config.RF_RANDOM_STATE)

    shap_idx = shap_rng.choice(len(X_test), size=shap_sample_size, replace=False)
    background_idx = shap_rng.choice(
        len(X_train),
        size=background_size,
        replace=False,
    )

    X_shap = X_test[shap_idx].astype(np.float64)
    y_shap = y_test[shap_idx]

    X_background = X_train[background_idx].astype(np.float64)

    if X_shap.shape[1] != rf.n_features_in_:
        raise ValueError(
            f"SHAP input has {X_shap.shape[1]} features, "
            f"but model expects {rf.n_features_in_}."
        )

    explainer = shap.TreeExplainer(
        rf,
        data=X_background,
        feature_perturbation="interventional",
        model_output="raw",
    )

    shap_values = explainer.shap_values(
        X_shap,
        check_additivity=False,
    )

    shap_values = np.asarray(shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:20]

    shap_top_features = [
        {
            "feature": feature_names[int(idx)],
            "bit_index": int(idx),
            "mean_abs_shap": float(mean_abs_shap[idx]),
        }
        for idx in top_idx
    ]

    shap_json_path = os.path.join(models_dir, "rf_shap_top20_features.json")
    with open(shap_json_path, "w") as f:
        json.dump(shap_top_features, f, indent=2)
    print(f"    SHAP top features -> {shap_json_path}")

    # Save reusable SHAP data for later molecule-level explanations
    shap_data_path = os.path.join(models_dir, "rf_shap_analysis_data.npz")
    np.savez_compressed(
        shap_data_path,
        X_shap=X_shap,
        y_shap=y_shap,
        shap_values=shap_values,
        expected_value=np.asarray(explainer.expected_value, dtype=float),
        shap_sample_indices=shap_idx,
        feature_names=np.array(feature_names),
    )
    print(f"    SHAP analysis data -> {shap_data_path}")

    # SHAP summary plot
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )

    path = os.path.join(plots_dir, "rf_shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot -> {path}")

    # Y-randomization / Y-scrambling test
    n_permutations = 50
    scrambled_r2_scores = []

    print(f"  Starting Y-randomization with {n_permutations} iterations...")

    rng = np.random.default_rng(config.RF_RANDOM_STATE)

    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_train)

        scores = cross_val_score(
            rf,
            X_train,
            y_shuffled,
            cv=kf,
            scoring="r2",
            n_jobs=-1,
        )
        scrambled_r2_scores.append(float(np.mean(scores)))

    original_r2 = float(cv_results["mean_r2"])
    scrambled_mean = float(np.mean(scrambled_r2_scores))
    scrambled_std = float(np.std(scrambled_r2_scores))
    robust = bool(original_r2 > scrambled_mean + 2 * scrambled_std)

    y_randomization_results = {
        "n_permutations": n_permutations,
        "original_mean_cv_r2": original_r2,
        "scrambled_mean_r2_scores": scrambled_r2_scores,
        "scrambled_mean_r2": scrambled_mean,
        "scrambled_std_r2": scrambled_std,
        "robust_by_2std_rule": robust,
    }

    y_rand_path = os.path.join(models_dir, "rf_y_randomization.json")
    with open(y_rand_path, "w") as f:
        json.dump(y_randomization_results, f, indent=2)
    print(f"    Y-randomization results -> {y_rand_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scrambled_r2_scores, bins=15, alpha=0.7, label="Scrambled R2")
    ax.axvline(
        original_r2,
        color="red",
        linestyle="--",
        label=f"Original R2 ({original_r2:.2f})",
    )
    ax.set_title("Y-Randomization Test Results", fontsize=14)
    ax.set_xlabel("R2 Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()

    path = os.path.join(plots_dir, "rf_y_randomization.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot -> {path}")

    if robust:
        print("    SUCCESS: The model is statistically robust.")
    else:
        print("    WARNING: The model might be based on chance correlations.")

    # ── Save model ────────────────────────────────────────────
    # pyrefly: ignore [missing-import]
    import joblib

    model_path = os.path.join(models_dir, "rf_model.joblib")
    joblib.dump(rf, model_path)
    print(f"  Model → {model_path}")

    # Save model config (needed for prediction)
    model_config = {
        "target_col": target_col,
        "morgan_radius": config.MORGAN_RADIUS,
        "morgan_nbits": config.MORGAN_NBITS,
        "metrics": metrics,
        "cross_validation": cv_results,
        "y_randomization": y_randomization_results,
        "shap": {
            "summary_plot": os.path.join(plots_dir, "rf_shap_summary.png"),
            "top_features_json": shap_json_path,
            "analysis_data": shap_data_path,
            "feature_names": feature_names,
            "sample_size": int(shap_sample_size),
        },
        "applicability_domain_uncertainty": {
            "artifact_json": adu_json_path,
            "train_fps_npz": npz_path,
            "ad10_threshold": ad10_thresh,
            "test_coverage_at_target_alpha": test_coverage,
        },
    }
    config_path = os.path.join(models_dir, "rf_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"  Config → {config_path}")

    return model_path


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: RF model")
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--bioactivity_type", default=None)
    args = parser.parse_args()
    run(args.receptor, args.bioactivity_type)
