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
    prediction_arrays = []
    for batch in prediction_batches:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch)
        prediction_arrays.append(batch.detach().cpu().numpy().reshape(-1))
    
    y_pred_val = np.concatenate(prediction_arrays)
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
