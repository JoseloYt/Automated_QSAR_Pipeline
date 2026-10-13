# Automated QSAR Pipeline

A pipeline for automatically building QSAR (Quantitative Structure-Activity Relationship) models, starting from nothing more than the name of a biological receptor/target (e.g. `FGFR1` or `EGFR`). The pipeline handles pulling bioactivity data from ChEMBL, cleaning it up, running exploratory analysis, training a Random Forest model on molecular fingerprints and, optionally, a graph neural network model (Chemprop/MPNN), while also computing the applicability domain and uncertainty bands for the predictions.

It started out as thesis code, and was later refactored into a 5-step pipeline plus a prediction script so it could be run against any target without touching the code every time.

## General idea

You run:

```bash
python run_pipeline.py --receptor FGFR1
```

and that kicks off, in order:

1. **Fetch** raw bioactivity data from ChEMBL for that receptor.
2. **Preprocessing**: SMILES cleanup, deduplication, conversion to pIC50, active/inactive classification.
3. **EDA + Lipinski descriptors**, with plots and a statistical test (Mann-Whitney U).
4. **Random Forest model** on Morgan fingerprints, with cross-validation, SHAP, Y-randomization, applicability domain, and conformal prediction.
5. **Chemprop (MPNN) model**, optional (can be skipped with `--skip_chemprop` if you don't want to wait for the network to train or don't have Chemprop installed).

Each step saves its results to disk (`output/<RECEPTOR>/...`), so if a step fails you don't have to start over from scratch — you can just re-run that individual step's script.

Once models are trained, `predict.py` is used separately to predict bioactivity for new molecules (ones not used for training), with explainability (SHAP + substructure highlighting) and an indication of whether the molecule falls inside the model's applicability domain.

## Output folder structure

Everything the pipeline generates for a given receptor lands in `output/<RECEPTOR>/`:

```
output/<RECEPTOR>/
├── 01_raw_data.csv                # Step 1: raw ChEMBL data
├── 02_preprocessed.csv            # Step 2: cleaned SMILES + pIC50 + class
├── 03_eda_lipinski.csv            # Step 3: dataset enriched with descriptors
├── 03_mannwhitney_lipinski.json   # Step 3: statistical test results
├── pipeline.log                   # full run log
├── models/                        # trained models + metadata (Steps 4 and 5)
├── plots/                         # all plots (EDA, RF, Chemprop, AD/CP)
└── predictions/                   # outputs from predict.py on new molecules
```

## Modules

### `config.py`

The central configuration file. Anything I ever wanted to be able to tune without hunting for a magic number in the middle of the code lives here: default bioactivity type (`IC50`), organism (`Homo sapiens`), pIC50 thresholds for classifying active/inactive/intermediate (6.0 / 5.0), Morgan fingerprint parameters (radius 2, 2048 bits), the train/test split for the Random Forest, Chemprop training config (epochs, batch size), and the applicability domain (AD10) and conformal prediction settings (confidence level, minimum stratum size, etc.).

It also has a few helper functions (`get_output_dir`, `get_models_dir`, `get_plots_dir`) that build and create the output folders for a given receptor, so that `os.makedirs` call doesn't need to be repeated in every script.

### `step1_fetch_chembl.py`

Connects to ChEMBL via `chembl_webresource_client` and searches for targets matching the receptor name we pass in. Out of all the results, it keeps the one that's `Homo sapiens` + `SINGLE PROTEIN` with the best score (if none match that, it relaxes the filter and just keeps any human target). With the chosen `target_chembl_id`, it pulls all the biological activities of the requested type (IC50 by default), keeps only the ones with an exact relation (`=`, nothing like `>` or `<`), and saves the raw CSV with SMILES, bioactivity value, and units.

### `step2_preprocess.py`

This is where the heavy cleaning happens. It drops rows without SMILES or without a value, removes duplicates (by the SMILES as it comes straight from ChEMBL), strips salts off the molecules (keeps the longest fragment after splitting on the dot), validates and canonicalizes each SMILES with RDKit (if RDKit can't parse it, it's dropped), and deduplicates again, this time on the canonical SMILES.

It also clips absurdly high `standard_value` entries (above 100,000,000, which in practice are measurement junk) and, if the bioactivity type is IC50 or Ki, converts everything to pIC50 (`-log10(IC50 in M)`, assuming 1 nM as a floor to avoid taking the log of zero or a negative number). Once pIC50 is computed, it classifies each molecule as `active` (pIC50 ≥ 6), `inactive` (pIC50 < 5), or `intermediate` (in between) — these thresholds come from `config.py`. The file it saves (`02_preprocessed.csv`) is what all the following steps use.

### `step3_eda_lipinski.py`

Exploratory data analysis. Computes the four classic Lipinski descriptors (molecular weight, LogP, H-bond donors, H-bond acceptors) for every molecule using RDKit, and drops the ones RDKit couldn't process. Runs a Mann-Whitney U test comparing those descriptors between active and inactive molecules (using a pIC50 ≥ 6.5 cutoff for this specific comparison) to check for statistically significant differences, and saves the results as JSON.

It also generates a set of plots that give an initial feel for the dataset: distribution of the target (pIC50 or the raw value), frequency of each bioactivity class, boxplots of the Lipinski descriptors by class, the "chemical space" (MW vs. LogP colored by class), and a correlation heatmap between the target and the descriptors. All of this is saved to `plots/`, and the enriched dataset (with descriptors already computed) is saved as `03_eda_lipinski.csv`.

### `step4_rf_model.py`

The pipeline's "baseline" model: a Random Forest trained on Morgan fingerprints (radius and bit count defined in `config.py`). The flow here is a lot longer than in the earlier steps, because it doesn't just train the model — it also tries to characterize how trustworthy each prediction actually is:

- Generates the fingerprints, does an 80/20 split, and trains a `RandomForestRegressor` (default hyperparameters, 100 trees).
- Runs 5-fold cross-validation on the training set and reports mean R² and standard deviation.
- Evaluates on the test set (R², RMSE, MAE) and plots predicted vs. actual and the residual distribution.
- Computes the **applicability domain** via Tanimoto AD10 (average similarity to the 10 nearest neighbors in the training set): if a new molecule falls far away from everything the model saw during training, it gets flagged as "out of domain" and its prediction shouldn't be trusted as much.
- Computes the Random Forest's uncertainty (spread across trees, using out-of-bag predictions for calibration) and builds **conformal prediction intervals** (Mondrian-style, split by inside/outside the applicability domain), so the interval widens or narrows depending on how much the trees agree and how "familiar" the molecule is.
- Runs a **SHAP** analysis (`TreeExplainer`) to see which fingerprint bits (i.e. which substructures) push the prediction the most, and saves both the top-20 features and the raw data so individual molecules can be re-explained later (this gets reused in `predict.py`).
- Runs a **Y-randomization** test (50 permutations of the target, re-training with cross-validation each time): if the original model gets a notably better R² than models trained on a randomly shuffled target, that's a sign it isn't just learning from noise/chance.
- Saves the model (`rf_model.joblib`), all the configuration/metadata (`rf_config.json`), and the applicability domain + conformal calibration artifacts, so `predict.py` can load them later without having to retrain anything.

It relies heavily on `ad_uncertainty.py` for all the applicability domain and uncertainty logic (see below).

### `step5_chemprop_model.py`

The alternative, more "modern" model: a message-passing neural network (MPNN) using the [Chemprop](https://github.com/chemprop/chemprop) v2 library, which learns directly from the molecular graph instead of a fixed fingerprint. It's optional because Chemprop + PyTorch Lightning is a heavy dependency and training takes considerably longer than the Random Forest; that's why `run_pipeline.py` imports it lazily inside the function and there's a `--skip_chemprop` flag.

The flow: builds `MoleculeDatapoint` objects from the SMILES and target, splits 80/10/10 into train/val/test, normalizes the target, builds the model (`BondMessagePassing` + `MeanAggregation` + `RegressionFFN`), and trains it with PyTorch Lightning, saving the best checkpoint based on `val_loss`. Evaluates on the test set, saves metrics, and plots predicted vs. actual.

Just like the Random Forest, it also computes an applicability domain (using Morgan fingerprints just for that part, since Chemprop itself doesn't need them) and a conformal prediction interval calibrated on the validation set — with the caveat that the same validation set was used to pick the best checkpoint, so the reported coverage may be slightly optimistic (this is noted in the output JSON as a reminder).

### `step6_terpene_filter.py`

This step was added because of the scope of the study. Terpene molecules are underrepresented in bioactivity datasets, like ChEMBL. It filters the selected dataset by the top n molecules with the highest score of `terpene similarity`, `NP-likeness`, `terpene features`, and, optionally, similarity with the `target molecules` (the ones to be predicted). Then, the user may re-train a model with the new filtered dataset, so its applicability domain would cover terpene or natural molecules.

### `ad_uncertainty.py`

Shared module with all the applicability domain and uncertainty logic, so the Random Forest and Chemprop use exactly the same criteria (instead of two slightly different versions of the same calculation). The key pieces:

- **AD10 (Tanimoto applicability domain)**: for each molecule, computes the average Tanimoto similarity to its 10 nearest neighbors in the training set. The cutoff threshold is the 5th percentile of that same distribution computed within the training set itself (leave-one-out). If a new molecule's AD10 falls below that threshold, it's considered out of domain.
- **Random Forest uncertainty**: uses out-of-bag predictions (each training sample evaluated only by the trees that didn't see it) for calibration, and the spread across the full ensemble of trees for the test set or new molecules.
- **Mondrian-style conformal prediction**: splits the nonconformity scores into two groups (inside/outside the applicability domain) and computes a separate calibration quantile for each, so the prediction interval width adapts to how "familiar" the molecule is, not just to the model's uncertainty.
- A bunch of plotting functions (AD10 distribution, Williams-style plot, predicted-vs-actual with confidence intervals, coverage calibration, AD-reliability heatmap, chemical space PCA, etc.) used by `step4`, `step5`, and `predict.py`.
- Functions to save/load the applicability domain and calibration "artifacts" (`*_ad_uncertainty.json` + `*_ad_train_fps.npz`), which is essentially how the model remembers what it needs to compare a new molecule against without retraining anything.

### `predict.py`

The script for using already-trained models on molecules you want to evaluate (new candidates, not seen during training). It takes a CSV with `Molecule` and `Canonical SMILES` columns, cleans and canonicalizes the SMILES the same way step 2 does, and runs:

- **Random Forest prediction**: point prediction, uncertainty (spread across trees), whether the molecule falls inside the applicability domain, and a conformal prediction interval.
- **Chemprop prediction** (if a checkpoint is available; can be skipped with `--skip_chemprop`), with the same kind of applicability domain and interval information.
- **Per-molecule explainability**: for the Random Forest, computes SHAP per molecule, identifies the most important fingerprint bits, and generates both a waterfall plot and an image of the molecule with the relevant substructures highlighted (red if they push the prediction up, blue if they push it down).
- Context plots: where the new molecules fall relative to the training set's applicability domain distribution, a ranking of how far inside or outside the domain each one is, and a PCA projection of chemical space (training set vs. new molecules).

Everything is saved to `output/<RECEPTOR>/predictions/`: a JSON with the full per-molecule detail, a summary CSV, and the plots in the `plots/` subfolder.

### `run_pipeline.py`

The orchestrator. This is the normal entry point for running everything end to end: it takes the receptor (required) and optionally the bioactivity type and the flag to skip Chemprop, sets up logging to file and console, and runs steps 1 through 5 in order, with per-step error handling (if a step fails, it's logged and the pipeline continues on to the final summary instead of stopping cold). At the end it prints a summary table with success/failure and timing for each step.

Each step can also be run on its own (`python step1_fetch_chembl.py --receptor FGFR1`, etc.) if you just want to reprocess one specific part without repeating the whole pipeline.

## How to run it

Full pipeline:

```bash
python run_pipeline.py --receptor FGFR1
```

Specifying the bioactivity type (defaults to `IC50`, set in `config.py`):

```bash
python run_pipeline.py --receptor FGFR1 --bioactivity_type IC50
```

Without training Chemprop (in case it's not installed, or you just don't want to wait):

```bash
python run_pipeline.py --receptor EGFR --skip_chemprop
```

Predicting on new molecules, once models are already trained for that receptor:

```bash
python predict.py --receptor FGFR1 --input_csv candidates.csv
```

## Dependencies

RDKit, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, shap, joblib, chembl_webresource_client, and (only if using step 5 / Chemprop) chemprop + pytorch + pytorch-lightning. There's no `requirements.txt` in the project yet — still need to add one.

## Things to keep in mind

- The pipeline assumes there's a SMILES column and a reasonable numeric bioactivity value; if ChEMBL reports activities as "% inhibition" instead of a concentration, the pIC50 conversion doesn't make sense (this is flagged as something to watch out for in `step2_preprocess.py`).
- The active/inactive thresholds (pIC50 6.0 / 5.0) are a reasonable but arbitrary convention — for a different receptor or project goal it might be worth adjusting them in `config.py`.
- The applicability domain and conformal prediction intervals aren't a "seal of approval" — they're a way of quantifying how similar a new molecule is to what the model saw during training, not an experimental verification.
- For Chemprop, the reported empirical coverage uses the validation set for calibration, so it may be slightly optimistic (since that same set was used to pick the best checkpoint).
