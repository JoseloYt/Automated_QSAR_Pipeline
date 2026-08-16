# Automated QSAR Pipeline

Pipeline para armar modelos QSAR (Quantitative Structure-Activity Relationship) de forma automática, partiendo únicamente del nombre de un receptor/target biológico (por ejemplo `FGFR1` o `EGFR`). El pipeline se encarga de bajar los datos de bioactividad desde ChEMBL, limpiarlos, hacer el análisis exploratorio, entrenar un modelo de Random Forest sobre fingerprints moleculares y, opcionalmente, un modelo de redes neuronales de grafos (Chemprop/MPNN), calculando además el dominio de aplicabilidad y bandas de incertidumbre para las predicciones.

Nació como el código de una tesis y después se refactorizó en un pipeline de 5 pasos + script de predicción para poder correrlo sobre cualquier target sin tener que tocar el código cada vez.

## Idea general

Uno corre:

```bash
python run_pipeline.py --receptor FGFR1
```

y eso dispara, en orden:

1. **Fetch** de la data cruda desde ChEMBL para ese receptor.
2. **Preprocesamiento**: limpieza de SMILES, deduplicación, conversión a pIC50, clasificación en activo/inactivo.
3. **EDA + descriptores de Lipinski**, con gráficos y un test estadístico (Mann-Whitney U).
4. **Modelo Random Forest** sobre fingerprints de Morgan, con validación cruzada, SHAP, Y-randomization, dominio de aplicabilidad y predicción conforme.
5. **Modelo Chemprop (MPNN)**, opcional (se puede saltar con `--skip_chemprop` si no querés esperar a que entrene la red o no tenés Chemprop instalado).

Cada paso deja sus resultados guardados en disco (`output/<RECEPTOR>/...`), así que si un paso falla no hay que repetir todo desde cero: se puede volver a correr el script individual de ese paso.

Una vez que hay modelos entrenados, `predict.py` se usa aparte para predecir bioactividad sobre moléculas nuevas (no las que se usaron para entrenar), con explicabilidad (SHAP + resaltado de subestructuras) e indicando si la molécula cae dentro del dominio de aplicabilidad del modelo.

## Estructura de carpetas de salida

Todo lo que genera el pipeline para un receptor queda en `output/<RECEPTOR>/`:

```
output/<RECEPTOR>/
├── 01_raw_data.csv                # Paso 1: data cruda de ChEMBL
├── 02_preprocessed.csv            # Paso 2: SMILES limpios + pIC50 + clase
├── 03_eda_lipinski.csv            # Paso 3: dataset enriquecido con descriptores
├── 03_mannwhitney_lipinski.json   # Paso 3: resultados del test estadístico
├── pipeline.log                   # log completo de la corrida
├── models/                        # modelos entrenados + metadata (Pasos 4 y 5)
├── plots/                         # todos los gráficos (EDA, RF, Chemprop, AD/CP)
└── predictions/                   # salidas de predict.py sobre moléculas nuevas
```

## Módulos

### `config.py`

El archivo de configuración central. Todo lo que en algún momento quise poder tunear sin andar buscando el número mágico en medio del código vive acá: tipo de bioactividad por default (`IC50`), organismo (`Homo sapiens`), umbrales de pIC50 para clasificar activo/inactivo/intermedio (6.0 / 5.0), parámetros del fingerprint de Morgan (radio 2, 2048 bits), el split de train/test para el Random Forest, la config de entrenamiento de Chemprop (épocas, batch size), y los parámetros del dominio de aplicabilidad (AD10) y de la predicción conforme (nivel de confianza, tamaño mínimo de cada estrato, etc.).

También tiene un par de funciones helper (`get_output_dir`, `get_models_dir`, `get_plots_dir`) que arman y crean las carpetas de salida para un receptor dado, para no repetir ese `os.makedirs` en cada script.

### `step1_fetch_chembl.py`

Se conecta a ChEMBL vía `chembl_webresource_client` y busca targets que matcheen el nombre del receptor que le pasamos. De todos los resultados, se queda con el que sea `Homo sapiens` + `SINGLE PROTEIN` con mejor score (si no encuentra ninguno así, afloja el filtro y se queda con cualquier target humano). Con el `target_chembl_id` elegido, trae todas las actividades biológicas del tipo pedido (por default IC50), se queda solo con las que tienen relación exacta (`=`, nada de `>` o `<`), y guarda el CSV crudo con SMILES, valor de bioactividad y unidades.

### `step2_preprocess.py`

Acá pasa la limpieza fuerte. Elimina filas sin SMILES o sin valor, saca duplicados (por SMILES tal cual viene de ChEMBL), le quita las sales a las moléculas (se queda con el fragmento más largo al partir por el punto), valida y canoniza cada SMILES con RDKit (si RDKit no puede parsearlo, se descarta), y vuelve a deduplicar ya sobre el SMILES canónico.

También recorta los valores de `standard_value` absurdamente altos (por encima de 100,000,000, que en la práctica son basura de medición) y, si el tipo de bioactividad es IC50 o Ki, convierte todo a pIC50 (`-log10(IC50 en M)`, asumiendo 1 nM como piso para evitar log de cero o negativos). Con el pIC50 ya calculado, clasifica cada molécula en `active` (pIC50 ≥ 6), `inactive` (pIC50 < 5) o `intermediate` (en el medio) — estos umbrales salen de `config.py`. El archivo que deja guardado (`02_preprocessed.csv`) es el que usan todos los pasos siguientes.

### `step3_eda_lipinski.py`

Análisis exploratorio. Calcula los cuatro descriptores clásicos de Lipinski (peso molecular, LogP, donores y aceptores de puentes de hidrógeno) para cada molécula con RDKit, y descarta las que RDKit no pudo procesar. Corre un test de Mann-Whitney U comparando esos descriptores entre moléculas activas e inactivas (con un umbral de pIC50 ≥ 6.5 para esta comparación específica) para ver si hay diferencias estadísticamente significativas, y guarda los resultados en JSON.

Genera además una serie de gráficos que sirven para tener una primera idea del dataset: distribución del target (pIC50 o el valor crudo), frecuencia de cada clase de bioactividad, boxplots de los descriptores de Lipinski por clase, el "espacio químico" (MW vs LogP coloreado por clase) y un heatmap de correlación entre el target y los descriptores. Todo esto se guarda en `plots/` y el dataset enriquecido (con los descriptores ya calculados) queda en `03_eda_lipinski.csv`.

### `step4_rf_model.py`

El modelo "de base" del pipeline: un Random Forest entrenado sobre fingerprints de Morgan (radio y cantidad de bits definidos en `config.py`). El flujo acá es bastante más largo que en los pasos anteriores porque no solo entrena el modelo, sino que también intenta caracterizar qué tan confiable es cada predicción:

- Genera los fingerprints, hace el split 80/20 y entrena un `RandomForestRegressor` (hiperparámetros por defecto, 100 árboles).
- Corre 5-fold cross-validation sobre el set de entrenamiento y reporta R² medio y desvío.
- Evalúa en el set de test (R², RMSE, MAE) y grafica predicho vs. real y la distribución de residuos.
- Calcula el **dominio de aplicabilidad** vía Tanimoto AD10 (similitud promedio a los 10 vecinos más cercanos en el set de entrenamiento): si una molécula nueva cae muy lejos de todo lo que el modelo vio en entrenamiento, se marca como "fuera de dominio" y no debería confiarse tanto en su predicción.
- Calcula la incertidumbre del RF (dispersión entre árboles, usando predicciones out-of-bag para la calibración) y construye **intervalos de predicción conforme** (Mondrian, separados por dentro/fuera del dominio de aplicabilidad), de forma que el intervalo se achique o agrande según qué tan de acuerdo estén los árboles y qué tan "familiar" es la molécula.
- Corre un análisis **SHAP** (`TreeExplainer`) para ver qué bits del fingerprint (o sea, qué subestructuras) más empujan la predicción, y guarda tanto el top-20 de features como los datos crudos para poder re-explicar moléculas puntuales después (esto se reutiliza en `predict.py`).
- Corre una prueba de **Y-randomization** (50 permutaciones del target, re-entrenando con cross-validation cada vez): si el modelo original saca un R² bastante mejor que los modelos entrenados con el target mezclado al azar, es una señal de que no está aprendiendo de puro ruido/casualidad.
- Guarda el modelo (`rf_model.joblib`), toda la configuración/metadata (`rf_config.json`) y los artefactos de dominio de aplicabilidad + calibración conforme, para que `predict.py` los pueda cargar después sin tener que re-entrenar nada.

Usa harto `ad_uncertainty.py` para toda la parte de dominio de aplicabilidad e incertidumbre (ver más abajo).

### `step5_chemprop_model.py`

El modelo alternativo, más "moderno": una red de pasaje de mensajes (MPNN) usando la librería [Chemprop](https://github.com/chemprop/chemprop) v2, que aprende directamente sobre el grafo molecular en lugar de un fingerprint fijo. Es opcional porque Chemprop + PyTorch Lightning es una dependencia pesada y el entrenamiento tarda bastante más que el Random Forest; por eso `run_pipeline.py` lo importa recién adentro de la función (import perezoso) y tiene la flag `--skip_chemprop`.

El flujo: arma los `MoleculeDatapoint` a partir de los SMILES y el target, separa 80/10/10 en train/val/test, normaliza el target, arma el modelo (`BondMessagePassing` + `MeanAggregation` + `RegressionFFN`), y lo entrena con PyTorch Lightning guardando el mejor checkpoint según `val_loss`. Evalúa en test, guarda métricas y grafica predicho vs. real.

Igual que el Random Forest, también calcula dominio de aplicabilidad (usando fingerprints de Morgan solo para esa parte, ya que Chemprop en sí no los necesita) y predicción conforme calibrada con el set de validación — con la salvedad de que ese mismo set de validación se usó para elegir el mejor checkpoint, así que la cobertura reportada puede ser un poco optimista (queda anotado en el JSON de salida para no olvidarlo).

### `ad_uncertainty.py`

Módulo compartido con toda la lógica de dominio de aplicabilidad e incertidumbre, para que el Random Forest y Chemprop usen exactamente el mismo criterio (y no dos versiones ligeramente distintas del mismo cálculo). Las piezas clave:

- **AD10 (dominio de aplicabilidad por Tanimoto)**: para cada molécula, se calcula la similitud de Tanimoto promedio contra sus 10 vecinos más cercanos en el set de entrenamiento. El umbral de corte es el percentil 5 de esa misma distribución calculada dentro del propio set de entrenamiento (leave-one-out). Si una molécula nueva tiene un AD10 por debajo de ese umbral, se considera fuera de dominio.
- **Incertidumbre del Random Forest**: usa las predicciones out-of-bag (cada muestra de entrenamiento evaluada solo por los árboles que no la vieron) para calibrar, y la dispersión entre todos los árboles del ensamble para el set de test o moléculas nuevas.
- **Predicción conforme tipo Mondrian**: separa las puntuaciones de no conformidad en dos grupos (dentro / fuera del dominio de aplicabilidad) y calcula un cuantil de calibración distinto para cada grupo, de forma que el ancho del intervalo de predicción se ajuste según qué tan "familiar" es la molécula, no solo según la incertidumbre del modelo.
- Un montón de funciones de graficado (distribución de AD10, gráfico tipo Williams, predicho-vs-actual con intervalos de confianza, calibración de cobertura, heatmap de confiabilidad por dominio, PCA del espacio químico, etc.) que usan tanto `step4` como `step5` y `predict.py`.
- Funciones para guardar/cargar los "artefactos" de dominio de aplicabilidad y calibración (`*_ad_uncertainty.json` + `*_ad_train_fps.npz`), que es como el modelo recuerda contra qué se tiene que comparar una molécula nueva sin tener que volver a entrenar nada.

### `predict.py`

El script para usar los modelos ya entrenados sobre moléculas que uno quiera evaluar (candidatos nuevos, no vistos en entrenamiento). Recibe un CSV con columnas `Molecule` y `Canonical SMILES`, limpia y canoniza los SMILES igual que en el paso 2, y corre:

- **Predicción con el Random Forest**: predicción puntual, incertidumbre (desvío entre árboles), si la molécula cae dentro del dominio de aplicabilidad, e intervalo de predicción conforme.
- **Predicción con Chemprop** (si hay un checkpoint disponible; se puede saltar con `--skip_chemprop`), con el mismo tipo de información de dominio de aplicabilidad e intervalo.
- **Explicabilidad por molécula**: para el Random Forest, calcula SHAP por molécula, identifica los bits del fingerprint más importantes, y genera tanto un gráfico de cascada (waterfall) como una imagen de la molécula con las subestructuras resaltadas (rojo si empujan la predicción para arriba, azul si la empujan para abajo).
- Gráficos de contexto: dónde caen las moléculas nuevas respecto a la distribución de dominio de aplicabilidad del set de entrenamiento, un ranking de qué tan "dentro" o "fuera" de dominio está cada una, y una proyección PCA del espacio químico (train vs. moléculas nuevas).

Todo se guarda en `output/<RECEPTOR>/predictions/`: un JSON con el detalle completo por molécula, un CSV resumen, y los plots en la subcarpeta `plots/`.

### `run_pipeline.py`

El orquestador. Es el punto de entrada normal para correr todo de punta a punta: recibe el receptor (obligatorio) y opcionalmente el tipo de bioactividad y la flag para saltarse Chemprop, configura logging a archivo y consola, y va ejecutando los pasos 1 a 5 en orden, con manejo de errores por paso (si un paso falla, lo registra y sigue con el resumen final en vez de cortar en seco). Al terminar imprime una tabla resumen con éxito/fallo y tiempo de cada paso.

Cada paso también se puede correr por separado (`python step1_fetch_chembl.py --receptor FGFR1`, etc.) si uno solo quiere reprocesar una parte puntual sin repetir todo el pipeline.

## Cómo correrlo

Pipeline completo:

```bash
python run_pipeline.py --receptor FGFR1
```

Especificando el tipo de bioactividad (por defecto usa `IC50`, definido en `config.py`):

```bash
python run_pipeline.py --receptor FGFR1 --bioactivity_type IC50
```

Sin entrenar Chemprop (por si no está instalado o simplemente no hace falta esperar):

```bash
python run_pipeline.py --receptor EGFR --skip_chemprop
```

Predicción sobre moléculas nuevas, una vez que ya hay modelos entrenados para ese receptor:

```bash
python predict.py --receptor FGFR1 --input_csv candidatos.csv
```

## Dependencias

RDKit, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, shap, joblib, chembl_webresource_client, y (solo si se usa el paso 5 / Chemprop) chemprop + pytorch + pytorch-lightning. No hay un `requirements.txt` todavía en el proyecto — pendiente de agregar.

## Cosas a tener en cuenta

- El pipeline asume que existe una columna de SMILES y un valor de bioactividad numérico razonable; si ChEMBL trae actividades reportadas como "% de inhibición" en vez de una concentración, la conversión a pIC50 no tiene sentido (queda comentado en `step2_preprocess.py` como algo a vigilar).
- Los umbrales de activo/inactivo (pIC50 6.0 / 5.0) son una convención razonable pero arbitraria — para otro receptor o otro objetivo de proyecto capaz conviene ajustarlos en `config.py`.
- El dominio de aplicabilidad y los intervalos de predicción conforme no son un "sello de garantía": son una forma de cuantificar qué tan parecida es una molécula nueva a lo que el modelo vio en entrenamiento, no una verificación experimental.
- Para Chemprop, la cobertura empírica reportada usa el set de validación como calibración, así que puede estar levemente optimista (ya que ese mismo set se usó para elegir el mejor checkpoint).
