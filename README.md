# Laboratorio II — Análisis de sentimientos (Sentiment140)

Sistema de análisis binario de sentimientos sobre Sentiment140, comparado experimentalmente en 3 etapas
(preprocesamiento → representación → clasificador), con trazabilidad completa en MLflow y despliegue en AWS.
Implementa el contrato definido en la guía **"Análisis de sentimientos"** (Laboratorio II - 2026 S02, PLN,
Universidad Sergio Arboleda) — ver [Importante/Laboratorio 2 - Análisis de Sentimientos.pdf](Importante/Laboratorio%202%20-%20Análisis%20de%20Sentimientos.pdf).

## Integrantes

Ver [MIEMBROS.md](MIEMBROS.md) para el mapeo `member_id` ↔ integrante ↔ Notebook Instance asignada.

- Julián Rincón (E01)
- Andrés Castro (E02)
- Juan Hurtado (E03)
- Miguel Flechas (E04)
- Paula Caballero (E05)

## Estado actual

**Infraestructura lista, experimentación pendiente.** Esto es lo que ya existe en la cuenta de AWS:

| Componente | Estado |
|---|---|
| MLflow Tracking Server | ✅ Corriendo en EC2 `mlflow` (`http://3.90.102.99:5000`), MLflow 3.1.4 verificado funcional |
| SageMaker Notebook Instances (una por integrante) | ✅ Creadas (`nlp-lab2-e01` a `nlp-lab2-e05`), rol `LabRole` |
| Experimento `nlp-lab2-sentiment140` en MLflow | ❌ Aún no creado — cero runs |
| Run de protocolo (dataset, muestra, folds) | ❌ Pendiente |
| T0 / B0 | ❌ Pendiente |
| Comparaciones de preprocesamiento/representación/clasificador | ❌ Pendiente |
| Ablación + configuración final | ❌ Pendiente |
| Modelo registrado (`sentiment140@champion`) | ❌ Pendiente |
| API FastAPI (`/api/v1/predict`, `/audit/*`, `/health`) | ❌ Pendiente |
| `reports/error_analysis.csv` / `.md` | ❌ Pendiente |

## Estructura

```
Actividad 2/
  Importante/           # guía oficial del laboratorio (PDF)
  MIEMBROS.md           # mapeo member_id / integrante / notebook_arn
  notebooks/            # experiment_audit.ipynb (sustentación)
  reports/              # error_analysis.csv y .md (se llenan al final)
  api/                  # FastAPI: /api/v1/predict, /audit/*, /health
  pipeline/             # lógica compartida: preprocesamiento, representación, clasificador
```

## Uso de inteligencia artificial generativa

Conforme a la sección 7 de la guía: se usa Claude Code (Anthropic) para diseño de arquitectura, automatización
de infraestructura AWS (SageMaker, MLflow, despliegue), implementación de la API y depuración. El clasificador
final se entrena por el equipo con los datos del laboratorio, sin APIs externas ni modelos preajustados para
la tarea. El equipo revisa el código y las recomendaciones antes de incorporarlas.
