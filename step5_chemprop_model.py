"""
Step 5 — Train a Chemprop MPNN regression model.

Replicates the logic from chemprop_regressor_fail(copia).ipynb:
  • Build MoleculeDatapoints from SMILES + targets
  • 80/20 random split (train/val)
  • SimpleMoleculeMolGraphFeaturizer
  • BondMessagePassing → MeanAggregation → RegressionFFN
  • Train with PyTorch Lightning
  • Evaluate on validation set
  • Save best checkpoint

Usage:
    python step5_chemprop_model.py --receptor FGFR1
"""

import argparse
import json
import os

import pandas as pd

import config


def run(receptor_name: str, bioactivity_type: str | None = None) -> str:
    """
    Main entry point for Step 5.
    Returns path to the best checkpoint.
    """
    # Lazy imports so the other steps still work if chemprop isn't installed
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from chemprop import data, featurizers, models, nn

    bio_type = bioactivity_type or config.BIOACTIVITY_TYPE
    out_dir = config.get_output_dir(receptor_name)
    models_dir = config.get_models_dir(receptor_name)
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

    # ── Split (80/20 train/val) ───────────────────────────────
    mols = [d.mol for d in all_data]
    split_sizes = config.CHEMPROP_SPLIT  # (0.8, 0.2)
    train_indices, val_indices, _ = data.make_split_indices(
        mols, config.CHEMPROP_SPLIT_TYPE, (split_sizes[0], split_sizes[1], 0.0)
    )
    train_data, val_data, _ = data.split_data_by_indices(
        all_data, train_indices, val_indices, None
    )
    print(f"  Train: {len(train_data[0])}  |  Val: {len(val_data[0])}")

    # ── Featurize ─────────────────────────────────────────────
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data[0], feat)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data[0], feat)
    val_dset.normalize_targets(scaler)

    # ── Dataloaders ───────────────────────────────────────────
    train_loader = data.build_dataloader(
        train_dset, num_workers=config.CHEMPROP_NUM_WORKERS
    )
    val_loader = data.build_dataloader(
        val_dset, num_workers=config.CHEMPROP_NUM_WORKERS, shuffle=False
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
    print("  Evaluating on validation set...")
    results = trainer.test(ckpt_path="best", dataloaders=val_loader, weights_only=False)
    # ── Save metrics ──────────────────────────────────────────
    metrics = results[0] if results else {}
    metrics_path = os.path.join(models_dir, "chemprop_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics → {metrics_path}")

    # Also save chemprop config info
    chemprop_config = {
        "target_col": target_col,
        "split": list(config.CHEMPROP_SPLIT),
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
