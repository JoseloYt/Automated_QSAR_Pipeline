"""
Predict bioactivity for new molecules with the trained RF and Chemprop models.

Expected input CSV columns:
  - Molecule
  - Canonical SMILES

Example:
    python predict.py --receptor FGFR1 --input_csv candidates.csv
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator
from PIL import Image, ImageDraw, ImageFont

import config
import ad_uncertainty as adu

warnings.filterwarnings("ignore", category=FutureWarning)


def strip_salts(smiles: str) -> str:
    """Keep the longest fragment, matching step2 preprocessing."""
    parts = str(smiles).split(".")
    return max(parts, key=len)


def validate_and_canonicalize(smiles: str) -> str | None:
    """Return canonical RDKit SMILES or None for invalid molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def preprocess_smiles(smiles: str) -> str | None:
    """Apply the same molecule cleaning used before RF training."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    return validate_and_canonicalize(strip_salts(str(smiles).strip()))


def smiles_to_morgan_fp(
    smiles: str,
    radius: int,
    n_bits: int,
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def safe_stem(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None and not pd.isna(value) else fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:80].strip("._") or fallback


def scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0])


def classify_prediction(prediction: float, target_col: str) -> str | None:
    if target_col != "pIC50":
        return None
    if prediction >= config.ACTIVE_THRESHOLD:
        return "active"
    if prediction < config.INACTIVE_THRESHOLD:
        return "inactive"
    return "intermediate"


def find_latest_chemprop_checkpoint(models_dir: Path) -> Path | None:
    ckpt_dir = models_dir / "chemprop_checkpoints"
    if not ckpt_dir.exists():
        return None

    best = sorted(ckpt_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime)
    if best:
        return best[-1]

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def predict_chemprop(checkpoint_path: Path, smiles: list[str]) -> list[float | None]:
    """Predict with the Chemprop v2 model saved by step5."""
    from lightning import pytorch as pl
    from chemprop import data, featurizers, models
    import torch

    datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer)
    loader = data.build_dataloader(dataset, shuffle=False, num_workers=0)

    model = models.MPNN.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
    )
    batches = trainer.predict(model, dataloaders=loader)
    if not batches:
        return [None] * len(smiles)

    tensors = []
    for batch in batches:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if not isinstance(batch, torch.Tensor):
            batch = torch.as_tensor(batch)
        tensors.append(batch.detach().cpu())

    preds = torch.cat(tensors, dim=0)
    return [float(x) for x in preds.numpy().reshape(-1)]


def bit_substructures(
    mol: Chem.Mol,
    bit_index: int,
    radius: int,
    n_bits: int,
) -> tuple[list[int], list[int], list[str]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    additional_output = rdFingerprintGenerator.AdditionalOutput()
    additional_output.AllocateBitInfoMap()
    generator.GetFingerprint(mol, additionalOutput=additional_output)
    bit_info = additional_output.GetBitInfoMap()

    atoms: set[int] = set()
    bonds: set[int] = set()
    fragments: list[str] = []

    for atom_idx, bit_radius in bit_info.get(int(bit_index), []):
        if bit_radius == 0:
            atoms.add(atom_idx)
            fragments.append(mol.GetAtomWithIdx(atom_idx).GetSymbol())
            continue

        env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, bit_radius, atom_idx)
        env_atoms = {atom_idx}
        for bond_idx in env_bonds:
            bond = mol.GetBondWithIdx(bond_idx)
            bonds.add(bond_idx)
            env_atoms.add(bond.GetBeginAtomIdx())
            env_atoms.add(bond.GetEndAtomIdx())

        atoms.update(env_atoms)
        try:
            fragments.append(
                Chem.MolFragmentToSmiles(
                    mol,
                    atomsToUse=sorted(env_atoms),
                    bondsToUse=list(env_bonds),
                    canonical=True,
                )
            )
        except Exception:
            pass

    unique_fragments = sorted({frag for frag in fragments if frag})
    return sorted(atoms), sorted(bonds), unique_fragments


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_highlighted_molecule(
    mol: Chem.Mol,
    bit_entries: list[dict[str, Any]],
    output_path: Path,
    title: str,
    legend: str,
) -> None:
    atoms: set[int] = set()
    bonds: set[int] = set()
    atom_colors: dict[int, tuple[float, float, float]] = {}
    bond_colors: dict[int, tuple[float, float, float]] = {}

    for entry in bit_entries:
        shap_value = float(entry.get("shap_value", 0.0))
        color = (0.9, 0.12, 0.10) if shap_value >= 0 else (0.10, 0.25, 0.95)
        entry_atoms = entry.get("highlight_atoms", [])
        entry_bonds = entry.get("highlight_bonds", [])
        atoms.update(entry_atoms)
        bonds.update(entry_bonds)
        for atom_idx in entry_atoms:
            atom_colors[int(atom_idx)] = color
        for bond_idx in entry_bonds:
            bond_colors[int(bond_idx)] = color

    image = Draw.MolToImage(
        mol,
        size=(900, 650),
        highlightAtoms=sorted(atoms),
        highlightBonds=sorted(bonds),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )

    title_height = 95
    legend_height = 150
    canvas = Image.new("RGB", (900, 650 + title_height + legend_height), "white")
    canvas.paste(image, (0, title_height))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34, bold=True)
    legend_font = load_font(25)

    display_title = title if len(title) <= 60 else f"{title[:57]}..."
    title_x = max((900 - text_width(draw, display_title, title_font)) // 2, 24)
    draw.text((title_x, 26), display_title, fill=(0, 0, 0), font=title_font)

    legend_lines = wrap_text(draw, legend, legend_font, max_width=840)
    y = 650 + title_height + 22
    for line in legend_lines[:4]:
        line_x = max((900 - text_width(draw, line, legend_font)) // 2, 24)
        draw.text((line_x, y), line, fill=(0, 0, 0), font=legend_font)
        y += 34

    canvas.save(output_path)


def save_waterfall_plot(
    values: np.ndarray,
    base_value: float,
    data_row: np.ndarray,
    feature_names: list[str],
    prediction: float,
    output_path: Path,
    max_display: int,
) -> None:
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=data_row,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(f"RF prediction: {prediction:.4f}", y=1.02)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def explain_rf_predictions(
    rf_model: Any,
    X: np.ndarray,
    clean_smiles: list[str],
    feature_names: list[str],
    shap_background: np.ndarray | None,
    plots_dir: Path,
    names: list[str],
    radius: int,
    n_bits: int,
    top_k: int,
    highlight_top_k: int,
) -> list[dict[str, Any]]:
    if shap_background is not None and len(shap_background):
        explainer = shap.TreeExplainer(
            rf_model,
            data=np.asarray(shap_background, dtype=np.float64),
            feature_perturbation="interventional",
            model_output="raw",
        )
    else:
        explainer = shap.TreeExplainer(rf_model, model_output="raw")

    shap_values = np.asarray(explainer.shap_values(X, check_additivity=False))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    base_value = scalar(explainer.expected_value)

    explanations: list[dict[str, Any]] = []

    for i, smiles in enumerate(clean_smiles):
        row_values = np.asarray(shap_values[i], dtype=float)
        present_bits = np.where(X[i] > 0)[0]
        ranked_bits = sorted(
            present_bits,
            key=lambda idx: abs(row_values[int(idx)]),
            reverse=True,
        )[:top_k]

        mol = Chem.MolFromSmiles(smiles)
        bit_entries: list[dict[str, Any]] = []
        for bit_idx in ranked_bits:
            atoms, bonds, fragments = bit_substructures(mol, int(bit_idx), radius, n_bits)
            shap_value = float(row_values[int(bit_idx)])
            bit_entries.append(
                {
                    "feature": feature_names[int(bit_idx)],
                    "bit_index": int(bit_idx),
                    "shap_value": shap_value,
                    "effect": "increases_prediction" if shap_value >= 0 else "decreases_prediction",
                    "fingerprint_value": int(X[i, int(bit_idx)]),
                    "substructure_smiles": fragments,
                    "highlight_atoms": atoms,
                    "highlight_bonds": bonds,
                }
            )

        stem = safe_stem(names[i], f"molecule_{i + 1}")
        waterfall_path = plots_dir / f"{i + 1:04d}_{stem}_rf_waterfall.png"
        highlight_path = plots_dir / f"{i + 1:04d}_{stem}_morgan_highlights.png"
        highlighted_entries = bit_entries[:highlight_top_k]

        if highlighted_entries:
            top_entry = highlighted_entries[0]
            fragments = top_entry.get("substructure_smiles") or ["substructure unavailable"]
            legend = (
                f"Highlighted substructure: {', '.join(fragments)}\n"
                f"SHAP value: {top_entry['shap_value']:.6f} "
                f"({top_entry['effect'].replace('_', ' ')})"
            )
        else:
            legend = "No highlighted Morgan substructure was found."

        save_waterfall_plot(
            values=row_values,
            base_value=base_value,
            data_row=X[i],
            feature_names=feature_names,
            prediction=float(rf_model.predict(X[i : i + 1])[0]),
            output_path=waterfall_path,
            max_display=max(top_k, 10),
        )
        draw_highlighted_molecule(
            mol,
            highlighted_entries,
            highlight_path,
            title=str(names[i]),
            legend=legend,
        )

        explanations.append(
            {
                "rf_expected_value": float(base_value),
                "rf_waterfall_plot": str(waterfall_path),
                "morgan_highlight_plot": str(highlight_path),
                "top_morgan_bits": [
                    {k: v for k, v in entry.items() if not k.startswith("highlight_")}
                    for entry in bit_entries
                ],
            }
        )

    return explanations


def load_rf_metadata(rf_config_path: Path, n_bits: int) -> tuple[str, list[str], np.ndarray | None]:
    target_col = "pIC50"
    feature_names = [f"bit_{i}" for i in range(n_bits)]
    shap_background = None

    if rf_config_path.exists():
        with open(rf_config_path, "r", encoding="utf-8") as f:
            rf_config = json.load(f)
        target_col = rf_config.get("target_col", target_col)
        feature_names = rf_config.get("shap", {}).get("feature_names", feature_names)
        shap_data = rf_config.get("shap", {}).get("analysis_data")
        if shap_data and Path(shap_data).exists():
            data = np.load(shap_data, allow_pickle=True)
            if "X_shap" in data:
                shap_background = np.asarray(data["X_shap"], dtype=np.float64)

    return target_col, feature_names, shap_background


def run(
    receptor: str,
    input_csv: str,
    output_json: str | None = None,
    output_csv: str | None = None,
    rf_model_path: str | None = None,
    chemprop_checkpoint: str | None = None,
    smiles_col: str = "Canonical SMILES",
    molecule_col: str = "Molecule",
    skip_chemprop: bool = False,
    top_k: int = 10,
    highlight_top_k: int = 1,
) -> str:
    out_dir = Path(config.get_output_dir(receptor))
    models_dir = Path(config.get_models_dir(receptor))
    predictions_dir = out_dir / "predictions"
    plots_dir = predictions_dir / "plots"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_csv)
    if output_json is None:
        output_path = predictions_dir / f"{input_path.stem}_predictions.json"
    else:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_csv is None:
        csv_output_path = predictions_dir / f"{input_path.stem}_predictions.csv"
    else:
        csv_output_path = Path(output_csv)
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    rf_path = Path(rf_model_path) if rf_model_path else models_dir / "rf_model.joblib"
    rf_config_path = models_dir / "rf_config.json"
    if not rf_path.exists():
        raise FileNotFoundError(f"Random Forest model not found: {rf_path}")

    # ── AD / uncertainty artifacts (per-model; see step4/step5) ─
    rf_ad_json_path = models_dir / "rf_ad_uncertainty.json"
    chemprop_ad_json_path = models_dir / "chemprop_ad_uncertainty.json"
    rf_ad = adu.load_ad_uncertainty_artifacts(str(rf_ad_json_path)) if rf_ad_json_path.exists() else None
    chemprop_ad = (
        adu.load_ad_uncertainty_artifacts(str(chemprop_ad_json_path))
        if chemprop_ad_json_path.exists() else None
    )
    if rf_ad is None:
        print("  Note: rf_ad_uncertainty.json not found; RF predictions will skip AD/CI reporting.")
    if chemprop_ad is None:
        print("  Note: chemprop_ad_uncertainty.json not found; Chemprop predictions will skip AD/CI reporting.")

    df = pd.read_csv(input_path)
    missing = [col for col in (molecule_col, smiles_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {missing}")

    radius = config.MORGAN_RADIUS
    n_bits = config.MORGAN_NBITS
    target_col, feature_names, shap_background = load_rf_metadata(rf_config_path, n_bits)

    rf_model = joblib.load(rf_path)

    records: list[dict[str, Any]] = []
    valid_rows: list[int] = []
    clean_smiles: list[str] = []
    names: list[str] = []
    fps: list[np.ndarray] = []

    for idx, row in df.iterrows():
        name = row[molecule_col]
        raw_smiles = row[smiles_col]
        cleaned = preprocess_smiles(raw_smiles)
        record = {
            "row_index": int(idx),
            "molecule": None if pd.isna(name) else str(name),
            "input_smiles": None if pd.isna(raw_smiles) else str(raw_smiles),
            "clean_smiles": cleaned,
            "valid": cleaned is not None,
        }
        records.append(record)

        if cleaned is None:
            record["error"] = "Invalid or empty SMILES after preprocessing."
            continue

        fp = smiles_to_morgan_fp(cleaned, radius=radius, n_bits=n_bits)
        if fp is None:
            record["valid"] = False
            record["error"] = "Could not generate Morgan fingerprint."
            continue

        valid_rows.append(len(records) - 1)
        clean_smiles.append(cleaned)
        names.append(str(name) if not pd.isna(name) else f"row_{idx}")
        fps.append(fp)

    X = np.asarray(fps, dtype=np.float64) if fps else np.empty((0, n_bits), dtype=np.float64)

    if len(X):
        rf_ens_mean, rf_ens_std = adu.rf_ensemble_mean_std(rf_model, X)
        rf_predictions = rf_ens_mean  # equivalent to rf_model.predict(X)

        rf_ad10 = rf_in_ad = rf_lower = rf_upper = rf_width = None
        if rf_ad is not None:
            rf_ad10 = adu.compute_ad10(
                X, rf_ad["train_fps"].astype(np.float64), k=rf_ad["ad_k"], exclude_self=False
            )
            rf_in_ad = rf_ad10 >= rf_ad["ad_threshold"]
            q = np.where(rf_in_ad, rf_ad["calibration"]["q_in"], rf_ad["calibration"]["q_out"])
            rf_lower, rf_upper, rf_width = adu.conformal_interval(
                rf_ens_mean, q, sigma=rf_ens_std, beta=rf_ad["beta"]
            )

            if rf_ad.get("train_ad10") is not None:
                adu.plot_new_molecules_ad10(
                    rf_ad["train_ad10"], rf_ad10, rf_ad["ad_threshold"],
                    plots_dir / "rf_new_molecules_ad10.png", "RF", names=names,
                )
            adu.plot_ad10_ranked(
                rf_ad10, names, rf_ad["ad_threshold"],
                plots_dir / "rf_new_molecules_ad10_ranked.png", "RF",
            )
            adu.plot_chemical_space_pca(
                rf_ad["train_fps"], X, rf_in_ad,
                plots_dir / "rf_new_molecules_chemical_space.png", "RF", names=names,
            )

        explanations = explain_rf_predictions(
            rf_model=rf_model,
            X=X,
            clean_smiles=clean_smiles,
            feature_names=feature_names,
            shap_background=shap_background,
            plots_dir=plots_dir,
            names=names,
            radius=radius,
            n_bits=n_bits,
            top_k=top_k,
            highlight_top_k=highlight_top_k,
        )

        for pos, record_idx in enumerate(valid_rows):
            pred = float(rf_predictions[pos])
            records[record_idx]["rf_prediction"] = pred
            records[record_idx]["rf_predicted_class"] = classify_prediction(pred, target_col)
            records[record_idx]["rf_uncertainty_tree_std"] = float(rf_ens_std[pos])
            if rf_ad is not None:
                records[record_idx]["rf_ad10"] = float(rf_ad10[pos])
                records[record_idx]["rf_in_applicability_domain"] = bool(rf_in_ad[pos])
                records[record_idx]["rf_prediction_interval_lower"] = float(rf_lower[pos])
                records[record_idx]["rf_prediction_interval_upper"] = float(rf_upper[pos])
                records[record_idx]["rf_prediction_interval_width"] = float(rf_width[pos])
            records[record_idx]["interpretation"] = explanations[pos]

    chemprop_path = None
    chemprop_error = None
    if not skip_chemprop and len(clean_smiles):
        chemprop_path = Path(chemprop_checkpoint) if chemprop_checkpoint else find_latest_chemprop_checkpoint(models_dir)
        if chemprop_path is None or not chemprop_path.exists():
            chemprop_error = "Chemprop checkpoint not found; Chemprop predictions were skipped."
        else:
            try:
                print(f"[Predict] Predict Chemprop model with checkpoint: {chemprop_path}")
                
                chemprop_predictions = predict_chemprop(chemprop_path, clean_smiles)

                chemprop_ad10 = chemprop_in_ad = chemprop_lower = chemprop_upper = chemprop_width = None
                if chemprop_ad is not None:
                    cp_radius = chemprop_ad.get("morgan_radius", radius)
                    cp_nbits = chemprop_ad.get("morgan_nbits", n_bits)
                    cp_fps, cp_fp_valid = adu.smiles_list_to_fp_matrix(
                        clean_smiles, radius=cp_radius, n_bits=cp_nbits
                    )
                    chemprop_ad10 = adu.compute_ad10(
                        cp_fps, chemprop_ad["train_fps"].astype(np.float64),
                        k=chemprop_ad["ad_k"], exclude_self=False,
                    )
                    chemprop_in_ad = chemprop_ad10 >= chemprop_ad["ad_threshold"]
                    q = np.where(
                        chemprop_in_ad,
                        chemprop_ad["calibration"]["q_in"],
                        chemprop_ad["calibration"]["q_out"],
                    )
                    preds_arr = np.array(
                        [p if p is not None else np.nan for p in chemprop_predictions], dtype=float
                    )
                    chemprop_lower, chemprop_upper, chemprop_width = adu.conformal_interval(
                        preds_arr, q, sigma=None
                    )

                    if chemprop_ad.get("train_ad10") is not None:
                        adu.plot_new_molecules_ad10(
                            chemprop_ad["train_ad10"], chemprop_ad10, chemprop_ad["ad_threshold"],
                            plots_dir / "chemprop_new_molecules_ad10.png", "Chemprop", names=names,
                        )
                    adu.plot_ad10_ranked(
                        chemprop_ad10, names, chemprop_ad["ad_threshold"],
                        plots_dir / "chemprop_new_molecules_ad10_ranked.png", "Chemprop",
                    )
                    adu.plot_chemical_space_pca(
                        chemprop_ad["train_fps"], cp_fps, chemprop_in_ad,
                        plots_dir / "chemprop_new_molecules_chemical_space.png", "Chemprop", names=names,
                    )

                for pos, record_idx in enumerate(valid_rows):
                    records[record_idx]["chemprop_prediction"] = chemprop_predictions[pos]
                    records[record_idx]["chemprop_predicted_class"] = (
                        classify_prediction(chemprop_predictions[pos], target_col)
                        if chemprop_predictions[pos] is not None
                        else None
                    )
                    if chemprop_ad is not None:
                        records[record_idx]["chemprop_ad10"] = float(chemprop_ad10[pos])
                        records[record_idx]["chemprop_in_applicability_domain"] = bool(chemprop_in_ad[pos])
                        if chemprop_predictions[pos] is not None:
                            records[record_idx]["chemprop_prediction_interval_lower"] = float(chemprop_lower[pos])
                            records[record_idx]["chemprop_prediction_interval_upper"] = float(chemprop_upper[pos])
                            records[record_idx]["chemprop_prediction_interval_width"] = float(chemprop_width[pos])
            except Exception as exc:
                chemprop_error = f"Chemprop prediction failed: {exc}"

    payload = {
        "metadata": {
            "receptor": receptor,
            "input_csv": str(input_path),
            "output_csv": str(csv_output_path),
            "target": target_col,
            "rf_model": str(rf_path),
            "rf_config": str(rf_config_path) if rf_config_path.exists() else None,
            "chemprop_checkpoint": str(chemprop_path) if chemprop_path else None,
            "chemprop_error": chemprop_error,
            "morgan_radius": radius,
            "morgan_nbits": n_bits,
            "n_input_rows": int(len(df)),
            "n_valid_molecules": int(len(valid_rows)),
            "plots_dir": str(plots_dir),
            "rf_ad_uncertainty_artifact": str(rf_ad_json_path) if rf_ad is not None else None,
            "chemprop_ad_uncertainty_artifact": str(chemprop_ad_json_path) if chemprop_ad is not None else None,
            "rf_ad10_threshold": rf_ad["ad_threshold"] if rf_ad is not None else None,
            "chemprop_ad10_threshold": chemprop_ad["ad_threshold"] if chemprop_ad is not None else None,
        },
        "predictions": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_rows = [
        {
            "Molecule": record.get("molecule"),
            "SMILES": record.get("input_smiles"),
            "Clean SMILES": record.get("clean_smiles"),
            "RF prediction": record.get("rf_prediction"),
            "RF AD10": record.get("rf_ad10"),
            "RF in AD": record.get("rf_in_applicability_domain"),
            "RF uncertainty (tree std)": record.get("rf_uncertainty_tree_std"),
            "RF interval lower": record.get("rf_prediction_interval_lower"),
            "RF interval upper": record.get("rf_prediction_interval_upper"),
            "Chemprop prediction": record.get("chemprop_prediction"),
            "Chemprop AD10": record.get("chemprop_ad10"),
            "Chemprop in AD": record.get("chemprop_in_applicability_domain"),
            "Chemprop interval lower": record.get("chemprop_prediction_interval_lower"),
            "Chemprop interval upper": record.get("chemprop_prediction_interval_upper"),
        }
        for record in records
    ]
    pd.DataFrame(csv_rows).to_csv(csv_output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(f"Prediction table saved to {csv_output_path}")
    if chemprop_error:
        print(chemprop_error)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict QSAR bioactivity for new molecules.")
    parser.add_argument("--receptor", required=True, help="Target receptor name, e.g. FGFR1.")
    parser.add_argument("--input_csv", required=True, help="CSV containing Molecule and Canonical SMILES columns.")
    parser.add_argument("--output_json", default=None, help="Optional JSON output path.")
    parser.add_argument("--output_csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--rf_model_path", default=None, help="Optional path to rf_model.joblib.")
    parser.add_argument("--chemprop_checkpoint", default=None, help="Optional path to a Chemprop .ckpt file.")
    parser.add_argument("--smiles_col", default="Canonical SMILES", help="SMILES column name.")
    parser.add_argument("--molecule_col", default="Molecule", help="Molecule/name column.")
    parser.add_argument("--skip_chemprop", action="store_true", help="Only run Random Forest predictions.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of RF/Morgan bits to report in JSON and SHAP waterfall per molecule.")
    parser.add_argument("--highlight_top_k", type=int, default=1, help="Number of RF/Morgan bits to highlight in the molecule image.")
    args = parser.parse_args()

    run(
        receptor=args.receptor,
        input_csv=args.input_csv,
        output_json=args.output_json,
        output_csv=args.output_csv,
        rf_model_path=args.rf_model_path,
        chemprop_checkpoint=args.chemprop_checkpoint,
        smiles_col=args.smiles_col,
        molecule_col=args.molecule_col,
        skip_chemprop=args.skip_chemprop,
        top_k=args.top_k,
        highlight_top_k=args.highlight_top_k,
    )


if __name__ == "__main__":
    main()