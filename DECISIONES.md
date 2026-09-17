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
