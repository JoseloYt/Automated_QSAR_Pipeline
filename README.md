# Pipeline Automatizado de QSAR (ChEMBL → Modelos Predictivos)

Este proyecto implementa un pipeline completo y automatizado para construir modelos de **QSAR** (Quantitative Structure-Activity Relationship) a partir de datos de bioactividad extraídos de **ChEMBL**. Dado el nombre de un receptor/diana biológica (por ejemplo `FGFR1`), el pipeline:

1. Descarga los datos de bioactividad desde ChEMBL.
2. Los limpia y preprocesa (incluyendo conversión a pIC50).
3. Realiza un análisis exploratorio (EDA) con descriptores de Lipinski.
4. Entrena un modelo de **Random Forest** sobre huellas moleculares de Morgan, con interpretabilidad SHAP.
5. Entrena opcionalmente un modelo de **red neuronal de grafos (Chemprop / MPNN)**.
6. Permite predecir la bioactividad de moléculas nuevas con los modelos entrenados.

Todo el flujo se puede ejecutar de un solo golpe con `run_pipeline.py`, o paso a paso ejecutando cada script individualmente.

---

## Índice

- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Configuración (`config.py`)](#configuración-configpy)
- [Descripción de cada script](#descripción-de-cada-script)
- [Cómo usar el pipeline](#cómo-usar-el-pipeline)
- [Archivos de salida generados](#archivos-de-salida-generados)
- [Predicción sobre moléculas nuevas](#predicción-sobre-moléculas-nuevas)
- [Notas y consideraciones](#notas-y-consideraciones)

---

## Estructura del proyecto

```
.
├── config.py                  # Configuración central de todo el pipeline
├── step1_fetch_chembl.py      # Paso 1: descarga de datos desde ChEMBL
├── step2_preprocess.py        # Paso 2: limpieza y preprocesado
├── step3_eda_lipinski.py      # Paso 3: EDA + descriptores de Lipinski
├── step4_rf_model.py          # Paso 4: modelo Random Forest + SHAP
├── step5_chemprop_model.py    # Paso 5: modelo Chemprop (MPNN)
├── predict.py                 # Predicción sobre moléculas nuevas
├── run_pipeline.py             # Orquestador que ejecuta todos los pasos
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

Cada receptor consultado genera su propia carpeta dentro de `output/`, de modo que se pueden procesar múltiples dianas sin que se pisen los resultados entre sí.

---

## Requisitos

El proyecto depende principalmente de:

- `pandas`, `numpy`, `scipy`
- `rdkit` (manipulación y validación de moléculas)
- `chembl_webresource_client` (acceso a la API de ChEMBL)
- `scikit-learn` (Random Forest, métricas, split de datos)
- `shap` (interpretabilidad del modelo Random Forest)
- `matplotlib`, `seaborn` (gráficos)
- `joblib` (serialización del modelo RF)
- `Pillow` (composición de imágenes en `predict.py`)
- `chemprop` (v2) y `lightning` + `torch` — **solo necesarios si se usa el Paso 5**

Instalación orientativa:

```bash
pip install pandas numpy scipy rdkit chembl_webresource_client scikit-learn \
            shap matplotlib seaborn joblib pillow

# Solo si vas a entrenar el modelo Chemprop:
pip install chemprop lightning torch
```

> Los imports de Chemprop/Lightning/Torch son "perezosos" (lazy imports dentro de la función `run()`), así que el resto del pipeline funciona aunque no tengas esas librerías instaladas, salvo que quieras ejecutar el Paso 5.

---

## Configuración (`config.py`)

Todos los parámetros ajustables del pipeline están centralizados en `config.py`, para no tener que tocar el código de cada paso:

| Parámetro | Descripción |
|---|---|
| `BIOACTIVITY_TYPE` | Tipo de bioactividad a descargar de ChEMBL (`IC50`, `Ki`, `EC50`, etc.) |
| `ORGANISM` | Organismo objetivo (por defecto `"Homo sapiens"`) |
| `TARGET_TYPE` | Tipo de diana ChEMBL (por defecto `"SINGLE PROTEIN"`) |
| `CONVERT_TO_PIC50` | Si se convierte el valor crudo a pIC50 (solo aplica a IC50/Ki) |
| `ACTIVE_THRESHOLD` / `INACTIVE_THRESHOLD` | Umbrales de pIC50 para clasificar como activo/inactivo/intermedio |
| `MORGAN_RADIUS` / `MORGAN_NBITS` | Parámetros de las huellas moleculares de Morgan para el RF |
| `RF_TEST_SIZE` / `RF_RANDOM_STATE` | Split y semilla para el Random Forest |
| `CHEMPROP_SPLIT` / `CHEMPROP_SPLIT_TYPE` | Split y tipo de partición para Chemprop |
| `CHEMPROP_MAX_EPOCHS` / `CHEMPROP_BATCH_SIZE` / `CHEMPROP_NUM_WORKERS` | Hiperparámetros de entrenamiento de Chemprop |
| `BASE_OUTPUT_DIR` | Carpeta base donde se guardan todos los resultados (`output/`) |

`config.py` también expone funciones helper (`get_output_dir`, `get_models_dir`, `get_plots_dir`) que crean automáticamente las carpetas de salida por receptor.

---

## Descripción de cada script

### `step1_fetch_chembl.py` — Descarga de datos

- Busca en ChEMBL las dianas que coincidan con el nombre del receptor (`search_target`).
- Selecciona automáticamente la mejor diana humana de tipo `SINGLE PROTEIN` según el score de ChEMBL (`select_best_target`), con fallback a cualquier diana humana si no hay coincidencia exacta.
- Descarga los datos de actividad biológica (`fetch_activities`) para el tipo de bioactividad indicado (por defecto `IC50`).
- Limpia los datos crudos (`clean_raw_data`): elimina filas sin valor o sin SMILES, convierte el valor a numérico y conserva solo mediciones con relación exacta (`=`).
- Guarda el resultado en `output/<receptor>/01_raw_data.csv`.

### `step2_preprocess.py` — Preprocesado

- Elimina filas con SMILES o valores faltantes.
- Elimina SMILES duplicados (se conserva la primera aparición).
- Elimina sales (`_strip_salts`): conserva el fragmento más largo al dividir el SMILES por `.`.
- Valida y canoniza cada SMILES con RDKit (`_validate_and_canonicalize`), descartando moléculas inválidas.
- Vuelve a eliminar duplicados tras la canonización.
- Recorta valores extremos (`norm_value`): cualquier valor mayor a 100,000,000 se acota a 10,000,000.
- Si el tipo de bioactividad es `IC50` o `Ki`, convierte el valor a **pIC50** (`_to_pic50 = -log10(valor en M)`).
- Clasifica cada molécula como `active` / `inactive` / `intermediate` según los umbrales de `config.py` (o mediante partición por mediana si no se convierte a pIC50).
- Guarda el resultado en `output/<receptor>/02_preprocessed.csv`.

### `step3_eda_lipinski.py` — EDA y descriptores de Lipinski

- Calcula los descriptores de Lipinski para cada molécula: peso molecular (MW), LogP, número de donadores de puente de hidrógeno y número de aceptores.
- Descarta moléculas cuyo SMILES no pueda ser interpretado por RDKit.
- Ejecuta pruebas estadísticas de **Mann-Whitney U** comparando los descriptores de Lipinski entre moléculas activas e inactivas (umbral de pIC50 = 6.5), guardando el resultado en `03_mannwhitney_lipinski.json`.
- Genera varios gráficos en `output/<receptor>/plots/`:
  - Distribución del valor objetivo (`target_distribution.png`)
  - Frecuencia de clases de bioactividad (`class_frequency.png`)
  - Boxplots de los descriptores de Lipinski por clase (`lipinski_boxplots.png`)
  - Espacio químico MW vs LogP (`chemical_space.png`)
  - Mapa de calor de correlaciones (`correlation_heatmap.png`)
- Guarda el dataset enriquecido en `output/<receptor>/03_eda_lipinski.csv`.

### `step4_rf_model.py` — Modelo Random Forest

- Genera huellas moleculares de Morgan (radio y nº de bits configurables) a partir de los SMILES limpios.
- Divide los datos en entrenamiento/prueba (80/20 por defecto).
- Realiza validación cruzada de 5 particiones (K-Fold) sobre el conjunto de entrenamiento y guarda los resultados en `rf_cross_validation.json`.
- Entrena un `RandomForestRegressor` (100 árboles).
- Calcula métricas de desempeño (R², RMSE, MAE) tanto en entrenamiento como en prueba.
- Genera gráficos de predicho vs. real y de distribución de residuos.
- Ejecuta un análisis de interpretabilidad **SHAP** (TreeExplainer) sobre una muestra del conjunto de prueba, identificando los 20 bits de la huella molecular más influyentes y generando un gráfico resumen (`rf_shap_summary.png`).
- Ejecuta una prueba de **Y-randomization** (50 permutaciones) para verificar que el modelo no se basa en correlaciones casuales.
- Serializa el modelo entrenado (`rf_model.joblib`) y toda la configuración/metadatos en `rf_config.json`, dentro de `output/<receptor>/models/`.

### `step5_chemprop_model.py` — Modelo Chemprop (MPNN)

- Construye `MoleculeDatapoint`s a partir de los SMILES y el valor objetivo.
- Divide los datos en entrenamiento/validación/prueba (80/10/10 por defecto, configurable en `config.py`).
- Featuriza las moléculas con `SimpleMoleculeMolGraphFeaturizer`.
- Define la arquitectura: `BondMessagePassing` → `MeanAggregation` → `RegressionFFN`.
- Entrena con PyTorch Lightning durante el número de épocas configurado, guardando el mejor checkpoint según `val_loss`.
- Evalúa el modelo sobre el conjunto de prueba y genera un gráfico de predicho vs. real.
- Guarda las métricas (`chemprop_metrics.json`) y la configuración de entrenamiento (`chemprop_config.json`) en `output/<receptor>/models/`.

> Este paso es opcional y puede omitirse con `--skip_chemprop` si no se dispone de las dependencias de Chemprop/Lightning/Torch, o si solo se necesita el modelo Random Forest.

### `predict.py` — Predicción sobre moléculas nuevas

Permite usar los modelos ya entrenados (RF y, opcionalmente, Chemprop) para predecir la bioactividad de moléculas nuevas a partir de un CSV de entrada.

- El CSV de entrada debe tener (por defecto) las columnas `Molecule` y `Canonical SMILES` (nombres configurables con `--molecule_col` y `--smiles_col`).
- Preprocesa cada SMILES igual que en el Paso 2 (eliminación de sales + canonización).
- Genera la huella de Morgan y predice con el modelo Random Forest (`rf_model.joblib`).
- Calcula la explicación SHAP de cada predicción individual, generando:
  - Un gráfico *waterfall* de SHAP por molécula.
  - Una imagen de la molécula con los átomos/enlaces resaltados según su contribución SHAP (rojo = aumenta la predicción, azul = la disminuye).
- Si no se indica `--skip_chemprop`, busca automáticamente el checkpoint de Chemprop más reciente y también genera predicciones con ese modelo.
- Clasifica cada predicción como `active` / `inactive` / `intermediate` según los umbrales configurados (solo si el target es pIC50).
- Guarda los resultados en un JSON detallado y en un CSV resumen, dentro de `output/<receptor>/predictions/`.

### `run_pipeline.py` — Orquestador

- Ejecuta en orden los pasos 1 a 5 (o 1 a 4 si se usa `--skip_chemprop`), importando cada módulo de forma perezosa.
- Cada paso se ejecuta de forma segura: si falla, se registra el error pero el pipeline continúa con el resto de pasos y genera un resumen final.
- Configura un logger que escribe simultáneamente en consola y en `output/<receptor>/pipeline.log`.
- Al finalizar, imprime un resumen con el estado (éxito/fallo) y el tiempo de ejecución de cada paso.

### `config.py` — Configuración central

Ya descrito en la sección [Configuración](#configuración-configpy) más arriba.

---

## Cómo usar el pipeline

### Ejecución completa (recomendado)

```bash
python run_pipeline.py --receptor FGFR1
```

Esto ejecuta los 5 pasos en orden para el receptor `FGFR1`, usando el tipo de bioactividad por defecto (`IC50`, definido en `config.py`).

### Especificando el tipo de bioactividad

```bash
python run_pipeline.py --receptor FGFR1 --bioactivity_type IC50
```

### Omitiendo el modelo Chemprop (solo Random Forest)

```bash
python run_pipeline.py --receptor EGFR --skip_chemprop
```

### Especificando un directorio de salida distinto (informativo)

```bash
python run_pipeline.py --receptor FGFR1 --output_dir mi_carpeta_resultados
```

### Ejecutar los pasos de forma individual

También se puede ejecutar cada script por separado, siempre que los archivos de entrada del paso anterior ya existan en `output/<receptor>/`:

```bash
python step1_fetch_chembl.py --receptor FGFR1
python step2_preprocess.py --receptor FGFR1
python step3_eda_lipinski.py --receptor FGFR1
python step4_rf_model.py --receptor FGFR1
python step5_chemprop_model.py --receptor FGFR1
```

---

## Archivos de salida generados

Todo se guarda dentro de `output/<RECEPTOR>/`:

```
output/<RECEPTOR>/
├── pipeline.log                       # Log completo de la ejecución
├── 01_raw_data.csv                    # Datos crudos de ChEMBL
├── 02_preprocessed.csv                # Datos limpios + pIC50 + clase
├── 03_eda_lipinski.csv                # Datos + descriptores de Lipinski
├── 03_mannwhitney_lipinski.json       # Resultados de pruebas estadísticas
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
        ├── 0001_<molécula>_rf_waterfall.png
        └── 0001_<molécula>_morgan_highlights.png
```

---

## Predicción sobre moléculas nuevas

Una vez entrenados los modelos (Pasos 4 y, opcionalmente, 5), se puede predecir la bioactividad de moléculas nuevas:

```bash
python predict.py --receptor FGFR1 --input_csv candidatos.csv
```

El CSV de entrada (`candidatos.csv`) debe tener, como mínimo, estas columnas:

| Molecule | Canonical SMILES |
|---|---|
| Compuesto_1 | CCOc1ccc2nc(...)... |
| Compuesto_2 | ... |

Opciones adicionales:

```bash
python predict.py \
    --receptor FGFR1 \
    --input_csv candidatos.csv \
    --output_json resultados.json \
    --output_csv resultados.csv \
    --skip_chemprop \
    --top_k 10 \
    --highlight_top_k 1
```

- `--skip_chemprop`: solo genera predicciones con el modelo Random Forest.
- `--top_k`: número de bits de la huella de Morgan a reportar por molécula.
- `--highlight_top_k`: número de bits a resaltar visualmente sobre la estructura molecular.
- `--smiles_col` / `--molecule_col`: permiten usar nombres de columna distintos a los de por defecto.

---

## Notas y consideraciones

- El pipeline crea automáticamente todas las carpetas necesarias (`output/<receptor>/`, `models/`, `plots/`, `predictions/`), por lo que no es necesario crearlas a mano.
- La conversión a pIC50 solo se aplica cuando el tipo de bioactividad es `IC50` o `Ki` **y** `CONVERT_TO_PIC50 = True` en `config.py`. Para otros tipos (por ejemplo `Inhibition`), la clasificación activo/inactivo se realiza mediante partición por mediana.
- El Paso 5 (Chemprop) requiere GPU o CPU con suficiente capacidad de cómputo, además de las librerías `chemprop`, `lightning` y `torch`. Si no se necesita este modelo, usar `--skip_chemprop` en `run_pipeline.py`.
- Los umbrales de clasificación (`ACTIVE_THRESHOLD`, `INACTIVE_THRESHOLD`) y los hiperparámetros de los modelos pueden ajustarse libremente en `config.py` sin tocar el resto del código.
- `predict.py` reutiliza el archivo `rf_config.json` generado en el Paso 4 (incluyendo el conjunto de fondo para SHAP), por lo que es importante haber ejecutado al menos el Paso 4 antes de predecir moléculas nuevas.
