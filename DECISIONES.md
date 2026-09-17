# Decisiones del equipo — selección de etapas (sección 3 de la guía)

Este documento registra las decisiones de selección exigidas por la sección 3: "Después seleccione
el preprocesamiento para la etapa siguiente" y "Con la representación seleccionada compare...".
Estas decisiones no quedan implícitas en ningún tag de MLflow — se documentan aquí explícitamente.

## Preprocesamiento seleccionado: **B0 (sin cambios)**

| Configuración | macro_f1_mean |
|---|---|
| B0 (referencia) | 0.7814 |
| P_ELONGATION | 0.7826 |
| P_LEMMA | 0.7794 |
| P_STOPWORDS_NEGATION | 0.7680 |
| P_STOPWORDS | 0.7585 |
| P_EMOJI | 0.7820 |

**Justificación:** la mejor alternativa (`P_ELONGATION`) solo mejora 0.0012 sobre B0 — dentro del
margen de desviación estándar observado entre folds (≈0.0006–0.0015) de todas las configuraciones.
No es una mejora significativa. El equipo decide **mantener el preprocesamiento de B0** como base
para la etapa de representación, tal como permite explícitamente la guía ("puede conservar B0").

## Representación seleccionada: **R_TFIDF_UNI_BI (TF-IDF, unigramas + bigramas)**

| Configuración | macro_f1_mean |
|---|---|
| R_TFIDF_UNI_BI | **0.7998** |
| R_TFIDF_UNI | 0.7879 |
| R_BOW (= B0) | 0.7814 |
| R_SPACY | 0.6778 |

**Justificación:** `R_TFIDF_UNI_BI` supera claramente a BoW (+1.84 puntos porcentuales), una mejora
real y consistente en los tres folds (std=0.0007), no atribuible a ruido. El equipo selecciona
**TF-IDF con unigramas y bigramas** como representación para la etapa de clasificador.

## Consecuencia: re-registro de la etapa de clasificador

Las comparaciones `C_LOGREG`, `C_LINEAR_SVM` y `C_SGD` originalmente registradas por Miguel (E04) y
Juan (E03) usaron BoW unigrama (representación de B0) porque se corrieron en paralelo, antes de que
esta selección quedara definida. Conforme al Anexo A.4 ("Todos los runs C_* usarán el mismo
preprocesamiento y la misma representación seleccionados"), se registran **3 runs nuevos**
`C_LOGREG`, `C_LINEAR_SVM`, `C_SGD` con `R_TFIDF_UNI_BI` como representación, ejecutados por Julián
(E01) desde su Notebook Instance, para cerrar la comparación correctamente sin alterar los runs
originales.

Los runs originales de Miguel y Juan (con BoW) **no se tocan ni se borran** — siguen siendo válidos
como parte de su contribución individual respectiva (el Anexo A.2 no exige que cada `lab_experiment_id`
sea único en el experimento; conviven ambas versiones). Los runs nuevos (con `R_TFIDF_UNI_BI`) son
los que se usan para la selección del pipeline candidato en la Fase 4.

## Pipeline candidato, ablación y revisión final

**Candidato** (tras las comparaciones obligatorias): `C_LOGREG` + `R_TFIDF_UNI_BI` + preprocesamiento
B0, run `e66a61d5fc9749549203feefdcffded6`, `macro_f1_mean=0.7998`. Difiere de B0 en una sola decisión
(`representation`), por lo que la guía exige exactamente una ablación (Sección 4).

**Ablación**: se revirtió `representation` a la forma de B0 (`bow`, `ngram_range=[1,1]`) manteniendo
fijo el resto. Run `7e71e6b0e8c74170b32cb248fcfe3118`, `macro_f1_mean=0.7814` (idéntico a B0, como se
espera al revertir exactamente esa decisión), `macro_f1_delta=0.0184`. Confirma que la mejora del
candidato proviene realmente de TF-IDF unigramas+bigramas y no de ruido experimental.

**Revisión final (`max_features=300000`)**: al reentrenar sobre los 1.360.000 registros completos de
train, un vocabulario TF-IDF sin acotar generó ~3.44M términos y agotó la memoria de la instancia
(`ml.t3.medium`, 4GB), matando el kernel silenciosamente. Se acotó el vocabulario a los 300.000
términos más frecuentes — práctica estándar de la industria para escalar TF-IDF a millones de
documentos con memoria limitada; es una decisión de la etapa de representación (`representation.
parameters.max_features`), no un cambio de clasificador ni de las etapas ya seleccionadas.

Como esto hace que la configuración final ya **no sea idéntica** a la del candidato original (Anexo
A.3: la igualdad se evalúa sobre todos los campos, incluidos `representation.parameters`), se siguió
el mecanismo explícito de la Sección 4 para este caso ("el equipo puede... elegir una configuración
revisada... evalúe esa configuración con los mismos tres folds y regístrela como EXTRA"):

1. Se evaluó la configuración revisada (`R_TFIDF_UNI_BI` + `max_features=300000` + `C_LOGREG`) con
   los mismos 3 folds del protocolo — run `EXTRA` `953498802691496697e0d8161ccd555a`,
   `lab_configuration_id=C_LOGREG_300K`, `macro_f1_mean=0.8006` (std=0.0005, ligeramente mejor que el
   candidato original: acotar el vocabulario también actúa como regularización).
2. El run final referencia este run `EXTRA` mediante `lab_selected_experiment_run_id`, y ambos
   comparten `lab_configuration_id=C_LOGREG_300K` con `run/configuration.json` exactamente
   equivalentes — verificado campo a campo.
