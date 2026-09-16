# Plan para retomar el Laboratorio II (cuenta individual nueva)

**Confirmado por el profesor:** cada integrante debe tener su propia cuenta de AWS Academy,
con **una** Notebook Instance (puede tener varios notebooks adentro). Nada de 5 instancias
SageMaker simultáneas en una sola cuenta — eso fue lo que disparó el guardrail dos veces.

Orden de ejecución revisado: todo lo que no es SageMaker se deja listo y verificado
primero; la única Notebook Instance (de Julián, E01) se crea **al final**, para
validar el pipeline end-to-end antes de indicarle al resto del equipo cómo replicarlo
cada uno en su propia cuenta.

## Fase 0 — Verificación (cuenta individual nueva, ya activa)

1. ✅ `aws sts get-caller-identity` confirma la cuenta nueva (`170100747321`), sin bloqueo `voc-cancel-cred`.
2. Taller 1 (EC2 + Lambda Function URL + API Gateway) redesplegado y verificado en esta cuenta — ver `Actividad 1/aws_apis/STATUS.md`.
3. **No** crear ninguna SageMaker Notebook Instance todavía — eso es la Fase 4, la última.

## Fase 1 — Arreglar el servidor MLflow (bloqueante) — ✅ RESUELTO

1. Instancia EC2 m5.large nueva (`i-0fb98819c3d73ff82`, `3.208.78.52`) en la cuenta individual.
2. Servidor MLflow 3.16.0 con `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////opt/mlflow/mlflow.db --serve-artifacts --artifacts-destination /opt/mlflow/artifacts --allowed-hosts "3.208.78.52,3.208.78.52:5000"`.
3. **Gotcha nuevo:** `--default-artifact-root <ruta local>` anula el proxy `mlflow-artifacts:/` y reproduce el mismo `PermissionError` de antes — hay que usar `--artifacts-destination` (no `--default-artifact-root`) para que los experimentos nuevos usen el esquema proxied.
4. **Gotcha nuevo (mlflow 3.x):** el servidor rechaza requests externas con `403 Invalid Host header` (protección DNS-rebinding) si no se pasa `--allowed-hosts` con la IP pública.
5. **Gotcha nuevo (2026-09-16):** con `--allowed-hosts` puesto pero sin `--cors-allowed-origins`, la UI web carga (GET) pero la **tabla de runs** (`#/experiments/<id>/runs`) queda en blanco con "Request error: INTERNAL_ERROR" — el POST a `/ajax-api/2.0/mlflow/runs/search` (y otros POST: `traces/metrics`, `search-datasets`, `logged-models/search`) devuelve `403 Forbidden` porque el origen `http://3.208.78.52:5000` no está en la lista por defecto (`localhost`/`127.0.0.1` solamente). Los runs individuales (`#/experiments/<id>/runs/<run_id>`) sí cargaban bien — solo la vista de lista fallaba, lo que hacía parecer que "no se veían los runs". Se agregó `--cors-allowed-origins "http://3.208.78.52:5000"` al `ExecStart` del systemd unit y se reinició; la tabla de runs ya carga los 15 runs correctamente.
6. Verificado con un run de prueba: param + métrica + artefacto → `FINISHED`, artefacto listado correctamente. Run de prueba borrado.
7. **Nota de navegación (mlflow 3.x):** al entrar a un experimento, la UI aterriza por defecto en "Overview" (pensado para trazas de LLM/GenAI). Para ver la tabla clásica de runs hay que ir explícitamente a la pestaña **Overview → (o URL `#/experiments/<id>/runs`)**; los demás tabs del menú (Traces, Sessions, Judges, Datasets, Evaluation runs, Playground, Prompts, Agent versions) son todos del flujo GenAI y no aplican a este laboratorio.

## Fase 2 — Re-registrar el protocolo (local, sin SageMaker todavía)

1. Correr de nuevo `pipeline/mlflow_logging.log_protocol_run(...)` con los archivos ya generados en `protocol/` (no hay que regenerarlos, ya están en el repo y son deterministas).
2. Confirmar en la UI de MLflow (`http://3.208.78.52:5000`) que el run de protocolo quedó `FINISHED` con sus dos artefactos.
3. Guardar el `protocol_run_id` real (se necesita para todos los runs siguientes vía `lab_protocol_run_id`).

## Fase 2 — ✅ RESUELTO

Run `protocol` registrado en `http://3.208.78.52:5000` (experimento `nlp-lab2-sentiment140`),
`run_id=0c141bf7df7e42289e25e68748f3a43d`, status `FINISHED`, ambos artefactos
(`protocol/members.csv`, `protocol/partitions.csv`) subidos correctamente vía el proxy.
Este es el `lab_protocol_run_id` real a usar en todos los runs experimentales siguientes.

**Gotcha encontrado y corregido:** `mlflow.log_param("cv_shuffle", True)` (booleano de Python)
se guarda como el string `"True"` — la guía exige exactamente `cv_shuffle=true` (minúscula).
Se corrigió a `mlflow.log_param("cv_shuffle", "true")` en `pipeline/mlflow_logging.py`, se borró
el run con el valor incorrecto y se volvió a registrar. Verificado parámetro por parámetro
contra el Anexo A.2 — los 8 params y los 2 artefactos coinciden exactamente.

## Fase 3 — Dry-run local de T0 y B0 (validar el pipeline end-to-end) — ✅ RESUELTO

Esto se corre **desde esta máquina o cualquier entorno de prueba**, NO desde
SageMaker todavía — el objetivo es confirmar que el código funciona antes de
gastar tiempo/recursos en las notebooks oficiales.

Corrido sobre una submuestra de 3.000 filas (de las 200.000 de `protocol/partitions.csv`,
mismos folds): T0 macro F1 ≈ 0.327 (esperado, baseline dummy), B0 macro F1 ≈ 0.706
(logistic regression + BOW, resultado sano). `preprocessing.py`, `representation.py`
y `classifier.py` corren sin errores. Runs de prueba **no** registrados en MLflow
(solo impresos en consola), tal como indica el punto 2 de abajo.

1. Entrenar T0 y B0 sobre uno o dos folds pequeños (o una submuestra) solo para
   confirmar que `pipeline/preprocessing.py`, `representation.py` y
   `classifier.py` no truenan y que las métricas se calculan bien.
2. **No registrar estos runs de prueba como oficiales** (o si se registran,
   usar `lab_experiment_id=EXTRA` claramente marcado como descartable, y
   borrarlos después) — los runs que cuentan deben venir de SageMaker con
   `notebook_arn` válido.
3. Revisar el tie-break de T0 (debe predecir `negative` en empate — `DummyClassifier`
   no lo garantiza, puede necesitar un ajuste manual, ver comentario en `classifier.py`).

## Fase 4 — Validación end-to-end con UNA sola SageMaker (Julián, E01)

Confirmado por el profesor: cada integrante tiene su **propia** cuenta de AWS Academy con
**una** Notebook Instance (no 5 en una cuenta compartida). Esta fase es solo para que Julián
valide que el pipeline completo funciona de punta a punta desde SageMaker real — el resto
del equipo repite este mismo procedimiento **cada uno en su propia cuenta** (Fase 4b), no aquí.

1. Crear **una única** Notebook Instance (`nlp-lab2-e01`) en la cuenta individual de Julián, rol `LabRole`.
2. Abrirla, copiar el código de `pipeline/` + `api/`, correr T0 y B0 con los 3 folds reales,
   registrar ambos runs con `lab_member_id=E01`, `notebook_arn` correcto y el
   `provenance/sagemaker-resource-metadata.json` real (copiado sin editar desde
   `/opt/ml/metadata/resource-metadata.json`).
3. Correr al menos una de las comparaciones obligatorias de preprocesamiento/representación/clasificador
   como prueba de que el circuito completo (SageMaker → MLflow → artefactos) funciona.
4. Con esto validado, **detener o eliminar la Notebook Instance** (no dejarla corriendo sin uso)
   y pasar a la Fase 4b.

## Fase 4b — Documentar para el resto del equipo (cada uno en su propia cuenta) — ✅ RESUELTO

1. ✅ Instrucciones paso a paso en [`GUIA_INTEGRANTES.md`](GUIA_INTEGRANTES.md) para que Andrés (E02),
   Juan (E03), Miguel (E04) y Paula (E05) repitan lo de la Fase 4 **cada uno en su propia
   cuenta individual de AWS Academy** — pedirle al profesor que los agregue si no la tienen.
   Cada uno apunta su MLflow al mismo tracking server (`http://3.208.78.52:5000`), no crea uno propio —
   basta con `MLFLOW_TRACKING_URI`, sin pasar por ninguna API intermedia; los resultados de todos se ven
   abriendo esa misma URL en el navegador.
2. Repartir las comparaciones obligatorias entre los 5 integrantes (mínimo 3
   configuraciones válidas cada uno, en al menos 2 etapas — ver sección 3 de
   la guía): preprocesamiento (`P_STOPWORDS`, `P_STOPWORDS_NEGATION`, `P_LEMMA`,
   `P_ELONGATION`, `P_EMOJI`), representación (`R_BOW`, `R_TFIDF_UNI`,
   `R_TFIDF_UNI_BI`, `R_SPACY`), clasificador (`C_LOGREG`, `C_LINEAR_SVM`, `C_SGD`).
3. Seleccionar pipeline candidato con base en validación cruzada (nunca con `test`).
4. Ablación: revertir a B0 cada decisión en que el candidato difiera (mínimo 1,
   mínimo 2 si difiere en ≥2 decisiones) — usar `pipeline.config.revert_decision`.
5. Reentrenar con las 1.360.000 filas completas de `train`, evaluar una sola vez
   sobre las 240.000 de `test`, registrar el run final con
   `lab_selected_experiment_run_id` apuntando al run experimental elegido.
6. Registrar el modelo en MLflow Model Registry: nombre `sentiment140`, alias `champion`.

## Fase 4c — QA de `api/main.py` contra el Anexo A.5 (2026-09-16) — ✅ RESUELTO

Corrida local (`uvicorn` en `127.0.0.1:8010`, apuntando al MLflow real) con un QA de 32 casos: los 6
endpoints, cada ejemplo inválido exacto del A.6 (`{}`, `text=null`, `""`, `"   "`, `[]`, `["ok",""]`,
`["ok",7]`, texto >1000 chars, lote >32), estructura exacta de cada respuesta, orden lexicográfico donde
la guía lo exige, y 8 solicitudes concurrentes. **32/32 PASS** tras corregir 2 bugs reales encontrados:

1. **`/audit/model` con `sentiment140@champion` inexistente devolvía `503 mlflow_unavailable` en vez de
   `404 champion_not_found`** — `_resolve_champion_model` envolvía la llamada con `_mlflow_call`, que
   intercepta *cualquier* `MlflowException` (incluida "no existe el alias") y la convierte en 503 antes
   de que el código pudiera distinguir "no existe" de "MLflow caído". Corregido: `_resolve_champion_model`
   ahora captura `MlflowException` directamente y solo devuelve 404 cuando `error_code ==
   "RESOURCE_DOES_NOT_EXIST"`, 503 en cualquier otro caso.
2. **`/audit/runs` tardaba 66-68 segundos** (el límite de la guía son 10s) — por cada run hacía una
   llamada `list_artifacts` para *comprobar si existe* `run/configuration.json` y otra `download_artifacts`
   para bajarlo, además de una tercera (recursiva) para el campo `artifacts` — 3 llamadas de red
   redundantes por run, en serie. Corregido: un solo listado recursivo de artefactos por run (reutilizado
   tanto para `artifacts` como para decidir si hay que descargar `configuration.json`), procesado en
   paralelo con `ThreadPoolExecutor` entre runs. Quedó en ~6s con los 15 runs actuales. También se subió
   el servidor MLflow de 4 a 8 `--workers` para dar más margen de concurrencia real cuando varios
   integrantes y el evaluador consulten a la vez.

## Fase 5 — Análisis de errores y despliegue

1. Muestrear ≥20 errores del modelo final (semilla 42), clasificar por categoría
   (A.6), generar `reports/error_analysis.csv` y `.md`, adjuntarlos como
   artefactos del run final.
2. Desplegar `api/main.py` (ya escrita y verificada localmente, ver Fase 4c) — falta el
   despliegue público (AWS Academy). Verificar los 6 endpoints contra el
   `sentiment140@champion` real una vez desplegada.
3. Confirmar accesibilidad externa de la API y del Tracking Server antes de
   entregar (sección 11 de la guía).

## Pendientes de decisión del equipo (no técnicos, hay que acordarlos)

- Lista definitiva de negadores para `P_STOPWORDS_NEGATION` (`pipeline/config.py`
  ya trae una propuesta en `NEGATORS_EN`, revisable).
- Modelo spaCy para `R_SPACY` (propuesto `en_core_web_md`, tiene vectores
  preentrenados — confirmar que sea el que el equipo quiere usar).
- Método de vector por documento (`document_vector_method`, propuesto
  `mean_token_vectors`).
- Cómo repartir exactamente las 3+ configuraciones por integrante para cumplir
  el mínimo individual sin duplicar trabajo.
