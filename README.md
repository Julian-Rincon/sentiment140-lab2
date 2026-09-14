# Laboratorio II — Análisis de sentimientos (Sentiment140)

Sistema de análisis binario de sentimientos sobre Sentiment140, comparado experimentalmente en 3 etapas
(preprocesamiento → representación → clasificador), con trazabilidad completa en MLflow y despliegue en AWS.
Implementa el contrato definido en la guía **"Análisis de sentimientos"** (Laboratorio II - 2026 S02, PLN,
Universidad Sergio Arboleda) — ver [Importante/Laboratorio 2 - Análisis de Sentimientos.pdf](Importante/Laboratorio%202%20-%20Análisis%20de%20Sentimientos.pdf).

Repositorio: **https://github.com/Julian-Rincon/sentiment140-lab2**

> **Para retomar el trabajo**, seguir [`PLAN.md`](PLAN.md) en orden — las SageMaker Notebook Instances
> se usan de último a propósito (ver sección "Bloqueos activos" abajo).

## Integrantes

Ver [MIEMBROS.md](MIEMBROS.md) para el mapeo `member_id` ↔ integrante ↔ Notebook Instance asignada.

- Julián Rincón (E01)
- Andrés Castro (E02)
- Juan Hurtado (E03)
- Miguel Flechas (E04)
- Paula Caballero (E05)

## Estado actual

| Componente | Estado |
|---|---|
| MLflow Tracking Server | ✅ Corriendo en EC2 m5.large `mlflow` (`http://3.208.78.52:5000`), `--serve-artifacts` + `--artifacts-destination` verificado con un run de prueba (artefacto subido, run `FINISHED`) |
| SageMaker Notebook Instance | ⏳ Se crea **una sola** (Julián, E01), al final, para validar el pipeline end-to-end — ver "Cambio de estrategia" abajo |
| Dataset (Sentiment140, revisión fija) | ✅ Cargado y verificado: 1.360.000 train / 240.000 test, coincide exacto con la guía |
| Muestra estratificada (200.000, semilla 42) + 3 folds | ✅ Generados en `protocol/partitions.csv` (folds balanceados 66.667/66.667/66.666) |
| `protocol/members.csv` | ✅ Generado con el mapeo de `MIEMBROS.md` |
| Experimento `nlp-lab2-sentiment140` en MLflow | ⏳ Se recrea en el servidor nuevo (el anterior quedó en una cuenta desactivada) |
| Código del pipeline (`pipeline/`: config, data, preprocessing, representation, classifier, mlflow_logging) | ✅ Escrito, probado localmente contra el dataset real |
| API FastAPI (`api/main.py`: `/api/v1/predict`, `/audit/*`, `/health`) | ✅ Escrita contra el contrato del Anexo A.5, aún sin desplegar |
| Run de protocolo registrado en MLflow (con artefactos) | ⏳ Pendiente de re-registrar en el servidor nuevo (ya no bloqueado) |
| T0 / B0 / comparaciones obligatorias / ablación / modelo final | ❌ Pendiente |
| `reports/error_analysis.csv` / `.md` | ❌ Pendiente |

### Cambio de estrategia — una SageMaker por cuenta individual, no 5 en una cuenta

Dos cuentas de AWS Academy seguidas se desactivaron (`voc-cancel-cred`) al crear **5 SageMaker Notebook Instances simultáneas** dentro de una misma cuenta — la segunda vez incluso espaciando las creaciones ~20-30 min, lo que confirmó que el disparador es la **cantidad concurrente**, no la velocidad de creación.

El profesor (Juan Pablo) confirmó por qué: **cada integrante del equipo debe tener su propia cuenta individual de AWS Academy**, y dentro de esa cuenta puede tener varios notebooks — no tiene sentido (ni el curso lo permite) que una sola cuenta sostenga 5 instancias SageMaker a la vez, "si esto fuera una cuenta empresarial, ¿sabes cuánto costaría eso?". Agregó a Julián a una cuenta nueva de forma individual, advirtiendo que si se vuelve a bloquear no podrá asignar otra.

**Estrategia revisada:**
1. Julián deja todo (Taller 1, servidor MLflow, código, protocolo) funcionando y verificado en su cuenta individual nueva.
2. Al final — y solo al final — crea **una única** Notebook Instance propia para validar el pipeline completo end-to-end (T0/B0 con los folds reales, registro en MLflow con `notebook_arn` y metadata real).
3. Con eso validado, se documenta el procedimiento exacto para que cada uno de los otros 4 integrantes lo repita **en su propia cuenta individual** (no en una compartida) — sección pendiente en este README una vez el paso 2 esté confirmado.

Ver [`PLAN.md`](PLAN.md) para el detalle fase por fase.

## Estructura

```
Actividad 2/
  Importante/           # guía oficial del laboratorio (PDF)
  MIEMBROS.md           # mapeo member_id / integrante / notebook_arn
  protocol/             # partitions.csv, members.csv (generados, ver pipeline/data.py)
  notebooks/            # experiment_audit.ipynb (sustentación) — pendiente
  reports/              # error_analysis.csv y .md (se llenan al final)
  api/main.py           # FastAPI: /api/v1/predict, /audit/*, /health
  pipeline/             # lógica compartida: config, data, preprocessing, representation, classifier, mlflow_logging
```

## Uso de inteligencia artificial generativa

Conforme a la sección 7 de la guía: se usa Claude Code (Anthropic) para diseño de arquitectura, automatización
de infraestructura AWS (SageMaker, MLflow, despliegue), implementación de la API y depuración. El clasificador
final se entrena por el equipo con los datos del laboratorio, sin APIs externas ni modelos preajustados para
la tarea. El equipo revisa el código y las recomendaciones antes de incorporarlas.
