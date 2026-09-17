# Laboratorio II — Análisis de sentimientos (Sentiment140)

Sistema de clasificación binaria de sentimientos (`negative` / `positive`) sobre el dataset **Sentiment140**, comparado experimentalmente en tres etapas (preprocesamiento → representación → clasificador), con trazabilidad completa en **MLflow** y despliegue en **AWS**. Implementa el contrato definido en la guía **"Análisis de sentimientos"** (Laboratorio II - 2026 S02, PLN, Universidad Sergio Arboleda) — ver [Importante/Laboratorio 2 - Análisis de Sentimientos.pdf](Importante/Laboratorio%202%20-%20Análisis%20de%20Sentimientos.pdf).

## Integrantes

Ver [MIEMBROS.md](MIEMBROS.md) para el mapeo `member_id` ↔ integrante ↔ Notebook Instance asignada (Anexo A.2).

- Julián Rincón (E01)
- Andrés Castro (E02)
- Juan Hurtado (E03)
- Miguel Flechas (E04)
- Paula Caballero (E05)

## Estado del proyecto

| Componente | Estado |
|---|---|
| Protocolo experimental (dataset, muestra, folds) | ✅ Fijado — 200.000 registros, semilla 42, 3 folds `StratifiedKFold` |
| Run de protocolo en MLflow (Anexo A.2) | ✅ Registrado, único, `FINISHED`, con `protocol/partitions.csv` y `protocol/members.csv` |
| MLflow Tracking Server | ✅ En producción, con `--serve-artifacts`, accesible públicamente |
| T0 / B0 y comparaciones obligatorias (los 5 integrantes) | ✅ Registradas, mínimo individual cumplido por los 5 (`valid_configurations>=3`, ≥2 etapas) |
| Pipeline candidato, ablación y configuración final | ✅ Candidato `R_TFIDF_UNI_BI` + `C_LOGREG`, ablación de representación evaluada (`macro_f1_delta=0.0184`), revisión final con `max_features=300000` registrada como `EXTRA` |
| Modelo final (`sentiment140@champion`) | ✅ Reentrenado con los 1.360.000 registros de train, evaluado una vez en test (`test_macro_f1=0.8244`), registrado en el Model Registry |
| `reports/error_analysis.csv` / `.md` | ✅ 30 errores muestreados (semilla 42), registrados como artefactos del run final |
| API FastAPI (`api/main.py`) | ✅ Implementada contra el Anexo A.5 y **desplegada públicamente** |
| `notebooks/experiment_audit.ipynb` | ✅ Reconstruye protocolo, runs y contribuciones consultando MLflow en vivo |

## Despliegue

| Servicio | URL |
|---|---|
| API pública | `http://3.208.78.52:8000` (`/health`, `/api/v1/predict`, `/audit/*`) |
| MLflow Tracking Server | `http://3.208.78.52:5000` |

Ambos corren en la misma instancia EC2 (`mlflow-taller2`, cuenta AWS Academy 170100747321) como servicios `systemd` independientes (`mlflow.service`, `sentiment-api.service`), gestionados vía AWS Systems Manager (sin exponer una clave SSH adicional).

El detalle de ejecución por fases está en [`PLAN.md`](PLAN.md). Instrucciones para que cada integrante corra
sus configuraciones asignadas desde su propia cuenta de AWS Academy están en
[`GUIA_INTEGRANTES.md`](GUIA_INTEGRANTES.md). Las decisiones de preprocesamiento/representación seleccionadas
y su justificación experimental están en [`DECISIONES.md`](DECISIONES.md).

## Arquitectura

```
Actividad 2/
  Importante/           # guía oficial del laboratorio (PDF)
  MIEMBROS.md           # mapeo member_id / integrante / notebook_arn (Anexo A.2)
  PLAN.md               # plan de ejecución por fases
  protocol/             # partitions.csv, members.csv — artefactos del run de protocolo
  pipeline/             # lógica compartida
    config.py               # configuraciones efectivas: T0, B0, comparaciones, ablación
    data.py                 # carga del dataset, muestra estratificada, folds
    preprocessing.py        # limpieza de texto (sección 3)
    representation.py       # BoW / TF-IDF / embeddings spaCy (sección 3)
    classifier.py            # construcción de clasificadores (DummyClassifier, LogReg, SVM, SGD)
    mlflow_logging.py       # registro de runs — nombres exactos de tags/params/métricas/artefactos del Anexo A.2
  api/main.py            # FastAPI: /api/v1/predict, /audit/*, /health (Anexo A.5)
  notebooks/             # experiment_audit.ipynb (sustentación)
  reports/               # error_analysis.csv / .md (análisis de errores del modelo final)
```

La lógica de MLflow, validación y estructura de proyecto reutiliza los patrones ya probados en el
[Laboratorio I](https://github.com/Julian-Rincon/nlp-pipeline-api) del equipo (validación estricta "todo o nada",
respuesta JSON con codificación UTF-8 explícita, separación entre lógica de dominio y capa HTTP).

## Protocolo experimental (sección 2 / Anexo A.2)

- **Dataset**: `adilbekovich/Sentiment140Twitter`, revisión fija `b6037e127257d95b9b23d31f78b264b9ebe697fd` (1.360.000 registros de entrenamiento / 240.000 de prueba).
- **Muestra**: 200.000 registros, muestreo estratificado, semilla 42.
- **Validación cruzada**: `StratifiedKFold`, 3 folds, `shuffle=true`, semilla 42.
- La muestra y los folds quedan fijados en un único run de protocolo (`lab_run_type=protocol`) del que dependen todas las comparaciones posteriores, referenciado desde cada run experimental mediante `lab_protocol_run_id`.

| Tipo de run | Contenido |
|---|---|
| **Protocolo** | Params exactos `dataset_id`, `dataset_revision`, `sampling_strategy`, `sample_size`, `random_seed`, `cv_strategy`, `cv_folds`, `cv_shuffle`; artefactos `protocol/partitions.csv` y `protocol/members.csv`. |
| **Experimental** (T0, B0, comparaciones, ablación) | Tags `lab_run_type=experiment`, `lab_protocol_run_id`, `lab_experiment_id`, `lab_stage`, `lab_member_id`, `lab_configuration_id`, `notebook_arn`; métricas `macro_f1_fold_0..2`, `macro_f1_mean`, `macro_f1_std`; artefactos `run/configuration.json` y `provenance/sagemaker-resource-metadata.json`. |
| **Final** | Tags `lab_run_type=final`, `lab_protocol_run_id`, `lab_selected_experiment_run_id`, `lab_configuration_id`, `lab_member_id`, `notebook_arn`; param `training_size=1360000`; artefactos de configuración, procedencia y análisis de errores. Registra el modelo en el Model Registry como `sentiment140`, alias `champion`. |

Cada integrante ejecuta sus runs experimentales desde su propio **SageMaker Notebook Instance** asignado por el curso — la procedencia (`provenance/sagemaker-resource-metadata.json`) se copia sin editar desde `/opt/ml/metadata/resource-metadata.json` de esa instancia.

## Comparaciones obligatorias (sección 3)

Cada integrante debe registrar al menos 3 configuraciones válidas en al menos 2 de las siguientes etapas, siempre partiendo de las decisiones de B0:

| Etapa | Configuraciones |
|---|---|
| Preprocesamiento | `P_STOPWORDS`, `P_STOPWORDS_NEGATION`, `P_LEMMA`, `P_ELONGATION`, `P_EMOJI` |
| Representación | `R_BOW`, `R_TFIDF_UNI`, `R_TFIDF_UNI_BI`, `R_SPACY` |
| Clasificador | `C_LOGREG`, `C_LINEAR_SVM`, `C_SGD` |

`R_SPACY` usa `en_core_web_md` (vectores preentrenados) con el vector de documento calculado como promedio de los vectores de token (`document_vector_method=mean_token_vectors`).

## API (Anexo A.5)

| Método y ruta | Contrato |
|---|---|
| `POST /api/v1/predict` | `{"text": string \| string[1..32]}` (máx. 1.000 caracteres por texto) → `{"model_run_id": string, "predictions": string[]}`, cada predicción `negative` o `positive`. Resuelve el modelo desde `sentiment140@champion`. |
| `GET /audit/protocol` | Datos del run de protocolo: `protocol_run_id`, `dataset_id`, `dataset_revision`, `sampling_strategy`, `sample_size`, `random_seed`, `cv_strategy`, `cv_folds`, `cv_shuffle`, `partitions_artifact`, `members_artifact`. `409 protocol_not_unique` si no hay exactamente un protocolo. |
| `GET /audit/runs` | Lista de todos los runs presentados (`run_id`, `status`, `run_type`, `params`, `metrics`, `tags`, `artifacts`, `configuration`), ordenada por `run_id`. |
| `GET /audit/contributions` | Contribución por integrante (`member_id`, `notebook_arn`, `run_ids`, `counted_run_ids`, `configuration_ids`, `stages`, `valid_configurations`), más `invalid_run_ids` y `unattributed_run_ids`. |
| `GET /audit/model` | Trazabilidad del modelo desplegado: `model_name=sentiment140`, `alias=champion`, `version`, `run_id`, `protocol_run_id`, `selected_experiment_run_id`, `configuration`, `training_size`, `test_macro_f1`. |
| `GET /health` | Resuelve `sentiment140@champion` y verifica inferencia; `200` si está disponible, `503` si no. |

Todos los endpoints JSON responden `application/json`. Los endpoints `/audit/*` consultan MLflow en tiempo de ejecución — si el Tracking Server no está disponible, responden `503 mlflow_unavailable` en vez de reconstruir la respuesta desde una copia local.

## Validación (Anexo A.5)

`/api/v1/predict` rechaza con `4xx`, sin resultados parciales: ausencia de `text`, `null`, string vacío o solo espacios, lista vacía, más de 32 elementos, textos de más de 1.000 caracteres, tipos distintos de string, o un lote con algún elemento inválido.

## Uso de inteligencia artificial generativa

Conforme a la sección 7 de la guía: se usa **Claude Code** (Anthropic) para diseño de arquitectura, automatización de infraestructura AWS (SageMaker, MLflow, despliegue), implementación de la API y depuración de errores. El clasificador final se entrena por el equipo con los datos del laboratorio, sin APIs externas ni modelos preajustados para la tarea — solo se permiten embeddings preentrenados como representación (`R_SPACY`). El equipo revisa el código y las recomendaciones antes de incorporarlas a la solución.

## Correr localmente

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

```bash
export MLFLOW_TRACKING_URI=http://<tracking-server>:5000
uvicorn api.main:app --reload
```

Docs interactivas en `http://127.0.0.1:8000/docs`.
