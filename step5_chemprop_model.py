"""
Step 5 — Train a Chemprop MPNN regression model.

Replicates the logic from chemprop_regressor_fail(copia).ipynb:
  • Build MoleculeDatapoints from SMILES + targets
    • 80/10/10 random split (train/val/test)
    • SimpleMoleculeMolGraphFeaturizer
    • BondMessagePassing → MeanAggregation → RegressionFFN
    • Train with PyTorch Lightning
    • Evaluate on test set
    • Save best checkpoint
              
Usage:
    python step5_chemprop_model.py --receptor FGFR1
"""
                  
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import config
import ad_uncertainty as adu
                  
                  
def _predictions_to_array(prediction_batches, torch) -> np.ndarray:
    """Flatten a list of chemprop/lightning prediction batches into a 1-D numpy array."""
    arrays = []
    for batch in prediction_batches:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch)
        arrays.append(batch.detach().cpu().numpy().reshape(-1))
    return np.concatenate(arrays)


def plot_pred_vs_actual(y_actual, y_pred, target_col: str, plots_dir: str) -> str:
    """Save a predicted-vs-actual plot for Chemprop test predictions."""
    r2 = r2_score(y_actual, y_pred)
                            
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_actual, y_pred, alpha=0.4, s=10)
    lo = min(y_actual.min(), y_pred.min())
    hi = max(y_actual.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_xlabel(f"Actual {target_col}", fontsize=12)
    ax.set_ylabel(f"Predicted {target_col}", fontsize=12)
    ax.set_title(
        f"Chemprop: Predicted vs Actual  (R²={r2:.3f})", fontsize=14
    )
    path = os.path.join(plots_dir, "chemprop_pred_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {path}")
    return path
                                                                                          
                                                                                          
def run(receptor_name: str, bioactivity_type: str | None = None) -> str:
    """
    Main entry point for Step 5.
    Returns path to the best checkpoint.
    """
    # Lazy imports so the other steps still work if chemprop isn't installed
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from chemprop import data, featurizers, models, nn
    import torch

    bio_type = bioactivity_type or config.BIOACTIVITY_TYPE
    out_dir = config.get_output_dir(receptor_name)
    models_dir = config.get_models_dir(receptor_name)
    plots_dir = config.get_plots_dir(receptor_name)
    preproc_path = os.path.join(out_dir, "02_preprocessed.csv")
                                                                                                                                                    
    print(f"[Step 5] Chemprop MPNN model for '{receptor_name}'")

    # ── Load data ─────────────────────────────────────────────
    df = pd.read_csv(preproc_path)
    convert = config.CONVERT_TO_PIC50 and bio_type.upper() in ("IC50", "KI")
    target_col = "pIC50" if convert else "standard_value"
    print(f"  Loaded {len(df)} rows, target: {target_col}")

    smis = df["Smiles"].values
    ys = df[[target_col]].values  # 2-D array required by chemprop

    # ── Build datapoints ──────────────────────────────────────
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    # ── Split (80/10/10 train/val/test) ───────────────────────
    mols = [d.mol for d in all_data]
    configured_split = tuple(config.CHEMPROP_SPLIT)
    if len(configured_split) == 3:
        split_sizes = configured_split
    else:
        split_sizes = (
            configured_split[0],
            configured_split[1] / 2,
            configured_split[1] / 2,
            )
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols, config.CHEMPROP_SPLIT_TYPE, split_sizes
    )
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    print(
    f"  Train: {len(train_data[0])}  |  "
    f"Val: {len(val_data[0])}  |  Test: {len(test_data[0])}"
    )

    # ── Featurize ─────────────────────────────────────────────
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
                                                                                                                                                                                                                                                                                                                                            
    train_dset = data.MoleculeDataset(train_data[0], feat)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data[0], feat)
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_data[0], feat)
    
    y_val = np.array([d.y for d in test_data[0]], dtype=float).reshape(-1).copy()
    
    # ── Dataloaders ───────────────────────────────────────────
    train_loader = data.build_dataloader(
        train_dset, num_workers=config.CHEMPROP_NUM_WORKERS
    )
    val_loader = data.build_dataloader(
        val_dset, num_workers=config.CHEMPROP_NUM_WORKERS, shuffle=False
    )
    test_loader = data.build_dataloader(
        test_dset, num_workers=config.CHEMPROP_NUM_WORKERS, shuffle=False
    )
    # ── Model ─────────────────────────────────────────────────
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)

    metric_list = [
        nn.metrics.RMSE(),
        nn.metrics.MAE(),
        nn.metrics.R2Score(),
        nn.metrics.MSE(),
    ]

    mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)
    # ── Training ──────────────────────────────────────────────

    ckpt_dir = os.path.join(models_dir, "chemprop_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpointing = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=config.CHEMPROP_MAX_EPOCHS,
        callbacks=[checkpointing],
    )

    print(f"  Training for {config.CHEMPROP_MAX_EPOCHS} epochs...")
    trainer.fit(mpnn, train_loader, val_loader)
    # ── Evaluate ──────────────────────────────────────────────
    print("  Evaluating on test set...")
    results = trainer.test(
        model = mpnn,
        dataloaders=test_loader, 
        ckpt_path="best",
        weights_only = False)
    prediction_batches = trainer.predict(
        mpnn,
        dataloaders=test_loader,
        ckpt_path="best",
        weights_only = False,
        return_predictions=True,
    )
    y_pred_val = _predictions_to_array(prediction_batches, torch)
    #y_val = np.array([d.y for d in test_data[0]], dtype=float).reshape(-1)

    plot_pred_vs_actual(y_val, y_pred_val, target_col, plots_dir)
    # ── Save metrics ──────────────────────────────────────────
    metrics = results[0] if results else {}
    metrics_path = os.path.join(models_dir, "chemprop_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics → {metrics_path}")
    
    # Also save chemprop config info
    chemprop_config = {
        "target_col": target_col,
        "split": list(split_sizes),
        "max_epochs": config.CHEMPROP_MAX_EPOCHS,
        "checkpoint_dir": ckpt_dir,
    }
    cfg_path = os.path.join(models_dir, "chemprop_config.json")
    with open(cfg_path, "w") as f:
        json.dump(chemprop_config, f, indent=2)
    print(f"  Config → {cfg_path}")

    # ── Applicability Domain + Mondrian conformal prediction ───
    # (target_col here is the true test-set target; y_val/y_pred_val above are test)
    print("  Computing applicability domain (Tanimoto AD10) for Chemprop...")

    train_smis = smis[train_indices[0]]
    val_smis = smis[val_indices[0]]
    test_smis = smis[test_indices[0]]
    y_train_true = np.array([d.y for d in train_data[0]], dtype=float).reshape(-1)
    y_val_true = np.array([d.y for d in val_data[0]], dtype=float).reshape(-1)
    y_test_true = y_val  # test set, per the (confusingly named) variable above

    train_fps, train_fp_valid = adu.smiles_list_to_fp_matrix(
        train_smis, radius=config.MORGAN_RADIUS, n_bits=config.MORGAN_NBITS
    )
    val_fps, val_fp_valid = adu.smiles_list_to_fp_matrix(
        val_smis, radius=config.MORGAN_RADIUS, n_bits=config.MORGAN_NBITS
    )
    test_fps, test_fp_valid = adu.smiles_list_to_fp_matrix(
        test_smis, radius=config.MORGAN_RADIUS, n_bits=config.MORGAN_NBITS
    )
    if not train_fp_valid.all():
        print(f"    Warning: {(~train_fp_valid).sum()} training SMILES could not be "
              f"fingerprinted for AD purposes; excluding from the AD reference set.")
    train_fps_valid_only = train_fps[train_fp_valid]

    train_ad10_loo = adu.compute_ad10(
        train_fps_valid_only, train_fps_valid_only, k=config.AD10_K, exclude_self=True
    )
    ad10_thresh = adu.ad_threshold(train_ad10_loo, percentile=config.AD10_PERCENTILE)
    val_ad10 = adu.compute_ad10(val_fps, train_fps_valid_only, k=config.AD10_K, exclude_self=False)
    test_ad10 = adu.compute_ad10(test_fps, train_fps_valid_only, k=config.AD10_K, exclude_self=False)
    in_ad_val = val_ad10 >= ad10_thresh
    in_ad_test = test_ad10 >= ad10_thresh
    print(f"    AD10 threshold ({config.AD10_PERCENTILE}th pct of training LOO AD10): {ad10_thresh:.4f}")
    print(f"    Test set in-AD: {in_ad_test.sum()}/{len(in_ad_test)}")

    adu.plot_ad_distribution(
        train_ad10_loo, test_ad10, ad10_thresh,
        os.path.join(plots_dir, "chemprop_ad10_distribution.png"), "Chemprop",
    )

    # Chemprop predictions on train (fit-quality only) and val (CP calibration)
    train_pred_loader = data.build_dataloader(
        train_dset, num_workers=config.CHEMPROP_NUM_WORKERS, shuffle=False
    )
    train_pred_batches = trainer.predict(
        mpnn, dataloaders=train_pred_loader, ckpt_path="best", weights_only=False,
        return_predictions=True,
    )
    y_pred_train = _predictions_to_array(train_pred_batches, torch)

    val_pred_batches = trainer.predict(
        mpnn, dataloaders=val_loader, ckpt_path="best", weights_only=False,
        return_predictions=True,
    )
    y_pred_val_cal = _predictions_to_array(val_pred_batches, torch)

    # NOTE: the validation set was used for checkpoint selection (monitor="val_loss"),
    # so reusing it as the conformal calibration set introduces a mild, commonly-accepted
    # optimism in coverage rather than a strictly held-out calibration split.
    cal_scores = np.abs(y_val_true - y_pred_val_cal)  # no per-sample sigma for Chemprop -> plain |residual|
    cp_calibration = adu.mondrian_calibrate(
        cal_scores, in_ad_val, alpha=config.CP_ALPHA, min_stratum_size=config.CP_MIN_STRATUM_SIZE,
    )
    print(f"    Mondrian conformal calibration (alpha={config.CP_ALPHA}): {cp_calibration}")

    q_test = np.where(in_ad_test, cp_calibration["q_in"], cp_calibration["q_out"])
    lower_test, upper_test, width_test = adu.conformal_interval(y_pred_val, q_test, sigma=None)
    test_coverage = adu.empirical_coverage(y_test_true, lower_test, upper_test)
    print(f"    Empirical test coverage @ {1 - config.CP_ALPHA:.0%} target: {test_coverage:.3f}")

    coverage_rows = adu.coverage_sweep(
        cal_scores, in_ad_val, y_test_true, y_pred_val, in_ad_test,
        sigma_test=None, beta=0.0, alphas=config.CP_COVERAGE_SWEEP_ALPHAS,
        min_stratum_size=config.CP_MIN_STRATUM_SIZE,
    )

    # ── AD / uncertainty plots ──────────────────────────────────
    test_rmse = float(np.sqrt(np.mean((y_test_true - y_pred_val) ** 2)))
    std_residuals = (y_test_true - y_pred_val) / (test_rmse + 1e-9)
    adu.plot_ad_williams(
        test_ad10, std_residuals, ad10_thresh,
        os.path.join(plots_dir, "chemprop_ad10_williams.png"), "Chemprop",
    )
    adu.plot_pred_vs_actual_ci(
        y_train_true, y_pred_train, y_test_true, y_pred_val, lower_test, upper_test,
        target_col, os.path.join(plots_dir, "chemprop_pred_vs_actual_ci.png"), "Chemprop",
    )
    test_abs_error = np.abs(y_test_true - y_pred_val)
    adu.plot_metric_vs_error_binned(
        test_ad10, test_abs_error, "AD10 similarity",
        os.path.join(plots_dir, "chemprop_ad10_vs_error.png"), "Chemprop",
    )
    adu.plot_coverage_calibration(
        coverage_rows, os.path.join(plots_dir, "chemprop_coverage_calibration.png"), "Chemprop",
    )
    adu.plot_ad_reliability_heatmap(
        in_ad_test, test_abs_error, config.AD_RELIABILITY_ERROR_THRESHOLD,
        os.path.join(plots_dir, "chemprop_ad_reliability_heatmap.png"), "Chemprop",
    )
    adu.plot_width_vs_metric(
        width_test, test_ad10, "AD10 similarity",
        os.path.join(plots_dir, "chemprop_interval_width_vs_ad10.png"), "Chemprop",
    )

    npz_path, adu_json_path = adu.save_ad_uncertainty_artifacts(
        out_dir=models_dir,
        prefix="chemprop",
        train_fps=train_fps_valid_only,
        ad_k=config.AD10_K,
        ad_threshold_value=ad10_thresh,
        calibration=cp_calibration,
        beta=0.0,
        train_ad10=train_ad10_loo,
        extra_json={
            "target_col": target_col,
            "test_coverage_at_target_alpha": test_coverage,
            "coverage_sweep": coverage_rows,
            "morgan_radius": config.MORGAN_RADIUS,
            "morgan_nbits": config.MORGAN_NBITS,
            "calibration_set": "validation (used in checkpoint selection; mild optimism expected)",
        },
    )
    print(f"    AD/uncertainty artifacts -> {adu_json_path}  |  {npz_path}")

    best_ckpt = checkpointing.best_model_path or checkpointing.last_model_path
    print(f"  Best checkpoint → {best_ckpt}")
    print(checkpointing.best_model_path)
    return best_ckpt


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Chemprop model")
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--bioactivity_type", default=None)
    args = parser.parse_args()
    run(args.receptor, args.bioactivity_type)
