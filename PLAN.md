# Plan para retomar el Laboratorio II en cuanto reactiven la cuenta

Orden de ejecución. **Las SageMaker Notebook Instances se usan de último a propósito**
(por precaución: sospechamos que crear varias de golpe pudo disparar el guardrail
que desactivó la cuenta) — todo lo demás se prepara y valida antes de tocarlas.

## Fase 0 — Verificación al reactivar (sin tocar SageMaker)

1. Confirmar `aws sts get-caller-identity` y `aws ec2 describe-instances` funcionan de nuevo.
2. **No** arrancar ni entrar a las notebooks `nlp-lab2-e01..e05` todavía.
3. Confirmar que la instancia EC2 `mlflow` sigue `running` (ya vimos que aguantó la desactivación).

## Fase 1 — Arreglar el servidor MLflow (bloqueante)

1. SSH a la instancia `mlflow` (EC2 Instance Connect, como con la de Activity 1).
2. Ver cómo está arrancado el proceso de MLflow actualmente (`ps aux | grep mlflow`, revisar el systemd unit o el comando usado).
3. Reiniciarlo agregando `--serve-artifacts` (o configurar `--default-artifact-root` a algo accesible remotamente si prefieren S3 en vez de disco local). Sin esto, ningún run con artefactos se puede registrar desde fuera del servidor.
4. Verificar con un run de prueba mínimo (logear un param + un artefacto chico) que ya no falla.

## Fase 2 — Re-registrar el protocolo (local, sin SageMaker todavía)

1. Correr de nuevo `pipeline/mlflow_logging.log_protocol_run(...)` con los archivos ya generados en `protocol/` (no hay que regenerarlos, ya están en el repo y son deterministas).
2. Confirmar en la UI de MLflow (`http://3.90.102.99:5000`) que el run de protocolo quedó `FINISHED` con sus dos artefactos.
3. Guardar el `protocol_run_id` real (se necesita para todos los runs siguientes vía `lab_protocol_run_id`).

## Fase 3 — Dry-run local de T0 y B0 (validar el pipeline end-to-end)

Esto se corre **desde esta máquina o cualquier entorno de prueba**, NO desde
SageMaker todavía — el objetivo es confirmar que el código funciona antes de
gastar tiempo/recursos en las notebooks oficiales.

1. Entrenar T0 y B0 sobre uno o dos folds pequeños (o una submuestra) solo para
   confirmar que `pipeline/preprocessing.py`, `representation.py` y
   `classifier.py` no truenan y que las métricas se calculan bien.
2. **No registrar estos runs de prueba como oficiales** (o si se registran,
   usar `lab_experiment_id=EXTRA` claramente marcado como descartable, y
   borrarlos después) — los runs que cuentan deben venir de SageMaker con
   `notebook_arn` válido.
3. Revisar el tie-break de T0 (debe predecir `negative` en empate — `DummyClassifier`
   no lo garantiza, puede necesitar un ajuste manual, ver comentario en `classifier.py`).

## Fase 4 — Ejecución oficial desde SageMaker (última fase, con cuidado)

**Abrir las notebooks de a una, no las 5 al tiempo**, y espaciar las acciones
en el tiempo por si el guardrail de Academy es sensible a ráfagas de actividad.

1. Julián (E01) abre `nlp-lab2-e01`, copia el código de `pipeline/` + `api/`,
   corre T0 y B0 con los 3 folds reales, registra ambos runs con
   `lab_member_id=E01`, `notebook_arn` correcto y el `provenance/sagemaker-resource-metadata.json`
   real (copiado sin editar desde `/opt/ml/metadata/resource-metadata.json`).
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

## Fase 5 — Análisis de errores y despliegue

1. Muestrear ≥20 errores del modelo final (semilla 42), clasificar por categoría
   (A.6), generar `reports/error_analysis.csv` y `.md`, adjuntarlos como
   artefactos del run final.
2. Desplegar `api/main.py` (ya escrita) — verificar los 6 endpoints contra el
   `sentiment140@champion` real.
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
