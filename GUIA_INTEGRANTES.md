# Guía de ejecución para el equipo (E02–E05)

Instrucciones para que cada integrante corra sus configuraciones asignadas desde su propia cuenta de AWS
Academy y las registre en el MLflow compartido del equipo. El pipeline y el protocolo experimental ya están
validados (run de protocolo registrado, dry-run local de T0/B0 sin errores) — esta guía es solo el paso a paso
de configuración.

## 1. Cuenta de AWS Academy

Cada integrante necesita **su propia cuenta individual** de AWS Academy (no la cuenta compartida del equipo).
Si no la tienes, pídesela al profesor. Dentro de tu cuenta vas a crear **una sola** SageMaker Notebook Instance
— no hace falta ni se debe crear más de una (dos cuentas del equipo ya se bloquearon por tener 5 notebooks
simultáneas en una misma cuenta).

Crear la instancia (consola de AWS o CLI, cualquiera):
- Servicio: **SageMaker → Notebook instances → Create notebook instance**
- Nombre: `nlp-lab2-<tu member_id>` (ej. `nlp-lab2-e02`)
- Tipo: `ml.t3.medium`
- Rol de IAM: `LabRole` (ya existe en la cuenta de Academy)

Espera a que quede en estado **InService** (normalmente 5-10 minutos) y ábrela desde la consola (botón "Open
Jupyter" o "Open JupyterLab").

## 2. Copiar el código

Dentro de la notebook, abre una terminal (JupyterLab → File → New → Terminal) y clona el repositorio:

```bash
git clone https://github.com/Julian-Rincon/sentiment140-lab2.git
cd sentiment140-lab2
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## 3. Configurar MLflow — solo necesitas la URL

El equipo comparte **un único** MLflow Tracking Server ya desplegado y funcionando. No hay que instalar ni
correr nada de MLflow por tu cuenta — solo apuntar el cliente a la URL:

```bash
export MLFLOW_TRACKING_URI=http://3.208.78.52:5000
```

(O en Python, al inicio de tu notebook: `mlflow.set_tracking_uri("http://3.208.78.52:5000")`.)

Con eso, cualquier `mlflow.start_run(...)` que corras desde tu notebook se registra directamente en el
servidor compartido — no hay API intermedia que llamar ni credenciales que configurar. Para **ver** los
resultados (tuyos y de los demás) en cualquier momento, simplemente abre esa misma URL en el navegador:
**http://3.208.78.52:5000** — ahí está la UI web de MLflow con todos los runs del experimento
`nlp-lab2-sentiment140`, filtrable por tags (`lab_member_id`, `lab_stage`, etc.).

## 4. Procedencia (obligatorio en cada run)

La guía exige que cada run experimental incluya el archivo de metadata real de tu Notebook Instance, sin
editar. Ya existe en cualquier notebook de SageMaker en esta ruta fija:

```bash
cp /opt/ml/metadata/resource-metadata.json ./provenance-metadata.json
```

Guárdalo así al inicio de tu notebook; lo vas a adjuntar como artefacto en cada run (`pipeline.mlflow_logging`
ya lo hace por ti si le pasas la ruta — ver el ejemplo abajo).

## 5. Correr tus configuraciones asignadas

Plantilla mínima para un run experimental (ajusta `member_id`, `notebook_arn`, `experiment_id_tag`, `stage` y
la función de configuración según tu asignación de la tabla de abajo):

```python
import sys; sys.path.insert(0, ".")
import mlflow, json
mlflow.set_tracking_uri("http://3.208.78.52:5000")

from pipeline import config
from pipeline.data import load_train_test
from pipeline.preprocessing import preprocess_texts
from pipeline.representation import build_vectorizer
from pipeline.classifier import build_classifier
from pipeline.mlflow_logging import log_experimental_run
from sklearn.metrics import f1_score
import pandas as pd

PROTOCOL_RUN_ID = "0c141bf7df7e42289e25e68748f3a43d"  # confirmar que sigue siendo el vigente en README.md
MEMBER_ID = "E02"                                      # tu member_id (ver MIEMBROS.md)
NOTEBOOK_ARN = "arn:aws:sagemaker:us-east-1:<tu-cuenta>:notebook-instance/nlp-lab2-e02"

train_split, _ = load_train_test()
train_df = train_split.to_pandas().reset_index().rename(columns={"index": "index"})
partitions = pd.read_csv("protocol/partitions.csv")
sample = partitions.merge(train_df[["index", "text", "label"]], on="index", how="left")

cfg = config.config_p_lemma()   # <- tu configuración asignada, ver tabla abajo

fold_scores = []
for fold_id in range(3):
    tr = sample[sample["fold"] != fold_id]
    va = sample[sample["fold"] == fold_id]
    train_texts = preprocess_texts(tr["text"].tolist(), cfg["preprocessing"])
    val_texts = preprocess_texts(va["text"].tolist(), cfg["preprocessing"])
    vec = build_vectorizer(cfg["representation"])
    X_tr, X_va = vec.fit_transform(train_texts), vec.transform(val_texts)
    clf = build_classifier(cfg["classifier"])
    clf.fit(X_tr, tr["label"])
    fold_scores.append(f1_score(va["label"], clf.predict(X_va), average="macro"))

run_id, mean_f1, std_f1 = log_experimental_run(
    run_name="P_LEMMA", experiment_id_tag="P_LEMMA", stage="preprocessing",
    member_id=MEMBER_ID, notebook_arn=NOTEBOOK_ARN, configuration_id="P_LEMMA",
    configuration=cfg, protocol_run_id=PROTOCOL_RUN_ID, fold_macro_f1=fold_scores,
    provenance_metadata_path="./provenance-metadata.json",
)
print(run_id, mean_f1, std_f1)
```

**Importante — el valor de `stage` es el `lab_stage` exacto que exige el Anexo A.4, no una descripción libre.**
Usa exactamente uno de estos strings (sensible a mayúsculas/minúsculas), según qué `experiment_id_tag` estés corriendo:

| `experiment_id_tag` | `stage` exacto |
|---|---|
| `T0` | `reference` |
| `B0` | `baseline` |
| `P_STOPWORDS`, `P_STOPWORDS_NEGATION`, `P_LEMMA`, `P_ELONGATION`, `P_EMOJI` | `preprocessing` |
| `R_BOW`, `R_TFIDF_UNI`, `R_TFIDF_UNI_BI`, `R_SPACY` | `representation` |
| `C_LOGREG`, `C_LINEAR_SVM`, `C_SGD` | `classifier` (en inglés, no "clasificador") |
| `ABLATION` | `ablation` |
| `EXTRA` | la etapa real a la que pertenezca (`preprocessing`, `representation`, `classifier` o `ablation`) |

Corre primero **T0 y B0** con tu propia notebook para confirmar que el circuito completo funciona antes de
tus configuraciones asignadas (usa `config.config_t0()` / `config.config_b0()` de la misma forma).

## 6. Avísale a Julián

Cuando tu Notebook Instance quede creada, manda el `notebook_arn` exacto (lo ves en la consola de SageMaker o
en `/opt/ml/metadata/resource-metadata.json`) para actualizar `protocol/members.csv` y `MIEMBROS.md` — es un
artefacto del run de protocolo y debe quedar con el ARN real de cada uno.

## Asignación de configuraciones obligatorias (sección 3 de la guía)

Cada integrante necesita mínimo 3 configuraciones válidas en al menos 2 etapas distintas (T0/B0 no cuentan).

| Integrante | member_id | Configuraciones asignadas | Etapas |
|---|---|---|---|
| Julián Rincón | E01 | `P_STOPWORDS`, `P_STOPWORDS_NEGATION`, `R_BOW` | preprocesamiento, representación |
| Andrés Castro | E02 | `P_LEMMA`, `P_ELONGATION`, `R_TFIDF_UNI` | preprocesamiento, representación |
| Juan Hurtado | E03 | `P_EMOJI`, `R_TFIDF_UNI_BI`, `R_SPACY` | preprocesamiento, representación |
| Miguel Flechas | E04 | `C_LOGREG`, `C_LINEAR_SVM`, + 1 configuración `EXTRA` (a definir en equipo) | clasificador, + 1 más |
| Paula Caballero | E05 | `C_SGD`, + 2 configuraciones `EXTRA` (a definir en equipo) | clasificador, + 1 más |

Las 12 comparaciones obligatorias (5 preprocesamiento + 4 representación + 3 clasificador) quedan cubiertas
entre Julián/Andrés/Juan. Miguel y Paula cubren los 3 clasificadores y completan su mínimo individual con
configuraciones `EXTRA` — el equipo las define juntos una vez estén los resultados de preprocesamiento y
representación (por ejemplo, combinar dos decisiones que ya funcionaron bien, o ajustar hiperparámetros del
clasificador seleccionado). La ablación (después de elegir el pipeline candidato) también cuenta como etapa
para el mínimo individual — es otra opción natural para Miguel y Paula si hace falta.

Función de configuración por comparación (`pipeline/config.py`):

| Código | Función |
|---|---|
| `P_STOPWORDS` | `config.config_p_stopwords()` |
| `P_STOPWORDS_NEGATION` | `config.config_p_stopwords_negation()` |
| `P_LEMMA` | `config.config_p_lemma()` |
| `P_ELONGATION` | `config.config_p_elongation()` |
| `P_EMOJI` | `config.config_p_emoji()` |
| `R_BOW` | `config.config_r_bow(preprocesamiento_seleccionado)` |
| `R_TFIDF_UNI` | `config.config_r_tfidf_uni(preprocesamiento_seleccionado)` |
| `R_TFIDF_UNI_BI` | `config.config_r_tfidf_uni_bi(preprocesamiento_seleccionado)` |
| `R_SPACY` | `config.config_r_spacy(preprocesamiento_seleccionado)` |
| `C_LOGREG` / `C_LINEAR_SVM` / `C_SGD` | `config.config_classifier(prep, rep, "logistic_regression" \| "linear_svm" \| "sgd")` |

`preprocesamiento_seleccionado` es el diccionario `preprocessing` de la configuración de preprocesamiento que
el equipo elija como base para la etapa de representación (ver sección 3 de la guía — por defecto, el de B0
si no se elige otro).
