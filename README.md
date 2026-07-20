# Automated QSAR Pipeline (ChEMBL → Predictive Models)

This project implements a complete, automated pipeline for building **QSAR** (Quantitative Structure-Activity Relationship) models from bioactivity data pulled from **ChEMBL**. Given the name of a receptor/biological target (e.g. `FGFR1`), the pipeline:

1. Downloads bioactivity data from ChEMBL.
2. Cleans and preprocesses it (including pIC50 conversion).
3. Runs exploratory data analysis (EDA) with Lipinski descriptors.
4. Trains a **Random Forest** model on Morgan fingerprints, with SHAP interpretability.
5. Optionally trains a **graph neural network (Chemprop / MPNN)** model.
6. Allows predicting bioactivity for new molecules using the trained models.

The whole flow can be run in one shot with `run_pipeline.py`, or step by step by running each script individually.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Configuration (`config.py`)](#configuration-configpy)
- [Script Descriptions](#script-descriptions)
- [How to Use the Pipeline](#how-to-use-the-pipeline)
- [Generated Output Files](#generated-output-files)
- [Predicting New Molecules](#predicting-new-molecules)
- [Notes and Considerations](#notes-and-considerations)

---

## Project Structure

```
.
├── config.py                  # Central configuration for the whole pipeline
├── step1_fetch_chembl.py      # Step 1: fetch data from ChEMBL
├── step2_preprocess.py        # Step 2: cleaning and preprocessing
├── step3_eda_lipinski.py      # Step 3: EDA + Lipinski descriptors
├── step4_rf_model.py          # Step 4: Random Forest model + SHAP
├── step5_chemprop_model.py    # Step 5: Chemprop (MPNN) model
├── predict.py                 # Prediction for new molecules
├── run_pipeline.py             # Orchestrator that runs every step
└── output/
    └── <RECEPTOR>/
        ├── 01_raw_data.csv
        ├── 02_preprocessed.csv
        ├── 03_eda_lipinski.csv
        ├── 03_mannwhitney_lipinski.json
        ├── pipeline.log
        ├── plots/
        ├── models/
        └── predictions/
```

Each queried receptor gets its own folder under `output/`, so multiple targets can be processed without results overwriting each other.

---

## Requirements

The project mainly depends on:

- `pandas`, `numpy`, `scipy`
- `rdkit` (molecule handling and validation)
- `chembl_webresource_client` (access to the ChEMBL API)
- `scikit-learn` (Random Forest, metrics, data splitting)
- `shap` (Random Forest model interpretability)
- `matplotlib`, `seaborn` (plots)
- `joblib` (RF model serialization)
- `Pillow` (image composition in `predict.py`)
- `chemprop` (v2) and `lightning` + `torch` — **only needed if Step 5 is used**

Suggested installation:

```bash
pip install pandas numpy scipy rdkit chembl_webresource_client scikit-learn \
            shap matplotlib seaborn joblib pillow

# Only if you're going to train the Chemprop model:
pip install chemprop lightning torch
```

> Chemprop/Lightning/Torch imports are "lazy" (imported inside the `run()` function), so the rest of the pipeline works even without those libraries installed, unless you want to run Step 5.

---

## Configuration (`config.py`)

Every tunable parameter of the pipeline is centralized in `config.py`, so you don't need to touch the code of each step:

| Parameter | Description |
|---|---|
| `BIOACTIVITY_TYPE` | Bioactivity type to fetch from ChEMBL (`IC50`, `Ki`, `EC50`, etc.) |
| `ORGANISM` | Target organism (default `"Homo sapiens"`) |
| `TARGET_TYPE` | ChEMBL target type (default `"SINGLE PROTEIN"`) |
| `CONVERT_TO_PIC50` | Whether to convert the raw value to pIC50 (only applies to IC50/Ki) |
| `ACTIVE_THRESHOLD` / `INACTIVE_THRESHOLD` | pIC50 thresholds for classifying active/inactive/intermediate |
| `MORGAN_RADIUS` / `MORGAN_NBITS` | Morgan fingerprint parameters used for the RF model |
| `RF_TEST_SIZE` / `RF_RANDOM_STATE` | Split and seed for Random Forest |
| `CHEMPROP_SPLIT` / `CHEMPROP_SPLIT_TYPE` | Split and split type for Chemprop |
| `CHEMPROP_MAX_EPOCHS` / `CHEMPROP_BATCH_SIZE` / `CHEMPROP_NUM_WORKERS` | Chemprop training hyperparameters |
| `BASE_OUTPUT_DIR` | Base folder where all results are saved (`output/`) |

`config.py` also exposes helper functions (`get_output_dir`, `get_models_dir`, `get_plots_dir`) that automatically create per-receptor output folders.

---

## Script Descriptions

### `step1_fetch_chembl.py` — Data Fetching

- Searches ChEMBL for targets matching the receptor name (`search_target`).
- Automatically selects the best human `SINGLE PROTEIN` target based on the ChEMBL score (`select_best_target`), falling back to any human target if there's no exact match.
- Downloads bioactivity data (`fetch_activities`) for the requested bioactivity type (default `IC50`).
- Cleans the raw data (`clean_raw_data`): drops rows with missing value or SMILES, converts the value to numeric, and keeps only measurements with an exact (`=`) relation.
- Saves the result to `output/<receptor>/01_raw_data.csv`.

### `step2_preprocess.py` — Preprocessing

- Drops rows with missing SMILES or values.
- Removes duplicate SMILES (keeps the first occurrence).
- Strips salts (`_strip_salts`): keeps the longest fragment when splitting the SMILES on `.`.
- Validates and canonicalizes each SMILES with RDKit (`_validate_and_canonicalize`), discarding invalid molecules.
- Removes duplicates again after canonicalization.
- Caps extreme values (`norm_value`): any value greater than 100,000,000 is clipped to 10,000,000.
- If the bioactivity type is `IC50` or `Ki`, converts the value to **pIC50** (`_to_pic50 = -log10(value in M)`).
- Classifies each molecule as `active` / `inactive` / `intermediate` based on the thresholds in `config.py` (or via a median split if not converting to pIC50).
- Saves the result to `output/<receptor>/02_preprocessed.csv`.

### `step3_eda_lipinski.py` — EDA and Lipinski Descriptors

- Computes Lipinski descriptors for every molecule: molecular weight (MW), LogP, number of hydrogen bond donors, and number of acceptors.
- Drops molecules whose SMILES can't be parsed by RDKit.
- Runs **Mann-Whitney U** statistical tests comparing Lipinski descriptors between active and inactive molecules (pIC50 threshold = 6.5), saving the result to `03_mannwhitney_lipinski.json`.
- Generates several plots in `output/<receptor>/plots/`:
  - Target value distribution (`target_distribution.png`)
  - Bioactivity class frequency (`class_frequency.png`)
  - Boxplots of Lipinski descriptors by class (`lipinski_boxplots.png`)
  - Chemical space MW vs LogP (`chemical_space.png`)
  - Correlation heatmap (`correlation_heatmap.png`)
- Saves the enriched dataset to `output/<receptor>/03_eda_lipinski.csv`.

### `step4_rf_model.py` — Random Forest Model

- Generates Morgan fingerprints (configurable radius and number of bits) from the cleaned SMILES.
- Splits the data into train/test sets (80/20 by default).
- Runs 5-fold cross-validation on the training set and saves the results to `rf_cross_validation.json`.
- Trains a `RandomForestRegressor` (100 trees).
- Computes performance metrics (R², RMSE, MAE) on both the training and test sets.
- Generates predicted-vs-actual and residual distribution plots.
- Runs a **SHAP** interpretability analysis (TreeExplainer) on a sample of the test set, identifying the 20 most influential fingerprint bits and producing a summary plot (`rf_shap_summary.png`).
- Runs a **Y-randomization** test (50 permutations) to verify the model isn't based on chance correlations.
- Serializes the trained model (`rf_model.joblib`) and all configuration/metadata to `rf_config.json`, inside `output/<receptor>/models/`.

### `step5_chemprop_model.py` — Chemprop (MPNN) Model

- Builds `MoleculeDatapoint`s from the SMILES and the target value.
- Splits the data into train/validation/test sets (80/10/10 by default, configurable in `config.py`).
- Featurizes the molecules with `SimpleMoleculeMolGraphFeaturizer`.
- Defines the architecture: `BondMessagePassing` → `MeanAggregation` → `RegressionFFN`.
- Trains with PyTorch Lightning for the configured number of epochs, saving the best checkpoint based on `val_loss`.
- Evaluates the model on the test set and generates a predicted-vs-actual plot.
- Saves the metrics (`chemprop_metrics.json`) and training configuration (`chemprop_config.json`) to `output/<receptor>/models/`.

> This step is optional and can be skipped with `--skip_chemprop` if you don't have the Chemprop/Lightning/Torch dependencies, or if you only need the Random Forest model.

### `predict.py` — Predicting New Molecules

Uses the already-trained models (RF and, optionally, Chemprop) to predict the bioactivity of new molecules from an input CSV.

- The input CSV must have (by default) the columns `Molecule` and `Canonical SMILES` (names configurable via `--molecule_col` and `--smiles_col`).
- Preprocesses each SMILES the same way as in Step 2 (salt stripping + canonicalization).
- Generates the Morgan fingerprint and predicts with the Random Forest model (`rf_model.joblib`).
- Computes a SHAP explanation for each individual prediction, generating:
  - A SHAP *waterfall* plot per molecule.
  - An image of the molecule with atoms/bonds highlighted according to their SHAP contribution (red = increases the prediction, blue = decreases it).
- Unless `--skip_chemprop` is passed, it automatically looks for the most recent Chemprop checkpoint and also generates predictions with that model.
- Classifies each prediction as `active` / `inactive` / `intermediate` based on the configured thresholds (only when the target is pIC50).
- Saves the results to a detailed JSON file and a summary CSV, inside `output/<receptor>/predictions/`.

### `run_pipeline.py` — Orchestrator

- Runs steps 1 through 5 in order (or 1 through 4 if `--skip_chemprop` is used), lazily importing each module.
- Each step runs safely: if it fails, the error is logged but the pipeline continues with the remaining steps and produces a final summary.
- Sets up a logger that writes simultaneously to the console and to `output/<receptor>/pipeline.log`.
- At the end, prints a summary with the status (success/failure) and execution time of each step.

### `config.py` — Central Configuration

Already described in the [Configuration](#configuration-configpy) section above.

---

## How to Use the Pipeline

### Full run (recommended)

```bash
python run_pipeline.py --receptor FGFR1
```

This runs all 5 steps in order for the `FGFR1` receptor, using the default bioactivity type (`IC50`, defined in `config.py`).

### Specifying the bioactivity type

```bash
python run_pipeline.py --receptor FGFR1 --bioactivity_type IC50
```

### Skipping the Chemprop model (Random Forest only)

```bash
python run_pipeline.py --receptor EGFR --skip_chemprop
```

### Specifying a different output directory (informational)

```bash
python run_pipeline.py --receptor FGFR1 --output_dir my_results_folder
```

### Running each step individually

You can also run each script separately, as long as the input files from the previous step already exist in `output/<receptor>/`:

```bash
python step1_fetch_chembl.py --receptor FGFR1
python step2_preprocess.py --receptor FGFR1
python step3_eda_lipinski.py --receptor FGFR1
python step4_rf_model.py --receptor FGFR1
python step5_chemprop_model.py --receptor FGFR1
```

---

## Generated Output Files

Everything is saved under `output/<RECEPTOR>/`:

```
output/<RECEPTOR>/
├── pipeline.log                       # Full execution log
├── 01_raw_data.csv                    # Raw ChEMBL data
├── 02_preprocessed.csv                # Clean data + pIC50 + class
├── 03_eda_lipinski.csv                # Data + Lipinski descriptors
├── 03_mannwhitney_lipinski.json       # Statistical test results
├── plots/
│   ├── target_distribution.png
│   ├── class_frequency.png
│   ├── lipinski_boxplots.png
│   ├── chemical_space.png
│   ├── correlation_heatmap.png
│   ├── rf_pred_vs_actual.png
│   ├── rf_residuals.png
│   ├── rf_shap_summary.png
│   ├── rf_y_randomization.png
│   └── chemprop_pred_vs_actual.png
├── models/
│   ├── rf_model.joblib
│   ├── rf_config.json
│   ├── rf_cross_validation.json
│   ├── rf_y_randomization.json
│   ├── rf_shap_top20_features.json
│   ├── rf_shap_analysis_data.npz
│   └── chemprop_checkpoints/
│       ├── best-*.ckpt
│       ├── last.ckpt
│       ├── chemprop_metrics.json
│       └── chemprop_config.json
└── predictions/
    ├── <input>_predictions.json
    ├── <input>_predictions.csv
    └── plots/
        ├── 0001_<molecule>_rf_waterfall.png
        └── 0001_<molecule>_morgan_highlights.png
```

---

## Predicting New Molecules

Once the models have been trained (Steps 4 and, optionally, 5), you can predict the bioactivity of new molecules:

```bash
python predict.py --receptor FGFR1 --input_csv candidates.csv
```

The input CSV (`candidates.csv`) must have, at minimum, these columns:

| Molecule | Canonical SMILES |
|---|---|
| Compound_1 | CCOc1ccc2nc(...)... |
| Compound_2 | ... |

Additional options:

```bash
python predict.py \
    --receptor FGFR1 \
    --input_csv candidates.csv \
    --output_json results.json \
    --output_csv results.csv \
    --skip_chemprop \
    --top_k 10 \
    --highlight_top_k 1
```

- `--skip_chemprop`: only generates predictions with the Random Forest model.
- `--top_k`: number of Morgan fingerprint bits to report per molecule.
- `--highlight_top_k`: number of bits to highlight visually on the molecular structure.
- `--smiles_col` / `--molecule_col`: allow using column names different from the defaults.

---

## Notes and Considerations

- The pipeline automatically creates all the necessary folders (`output/<receptor>/`, `models/`, `plots/`, `predictions/`), so there's no need to create them manually.
- pIC50 conversion only applies when the bioactivity type is `IC50` or `Ki` **and** `CONVERT_TO_PIC50 = True` in `config.py`. For other types (e.g. `Inhibition`), active/inactive classification is done via a median split.
- Step 5 (Chemprop) requires a GPU or a CPU with enough compute power, plus the `chemprop`, `lightning`, and `torch` libraries. If you don't need this model, use `--skip_chemprop` with `run_pipeline.py`.
- Classification thresholds (`ACTIVE_THRESHOLD`, `INACTIVE_THRESHOLD`) and model hyperparameters can be freely adjusted in `config.py` without touching the rest of the code.
- `predict.py` reuses the `rf_config.json` file generated in Step 4 (including the SHAP background set), so Step 4 must have been run at least once before predicting new molecules.
