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
| MLflow Tracking Server | ✅ Corriendo en EC2 `mlflow` (`http://3.90.102.99:5000`), MLflow 3.1.4 verificado funcional |
| SageMaker Notebook Instances (una por integrante) | ✅ Creadas (`nlp-lab2-e01` a `nlp-lab2-e05`), rol `LabRole` |
| Dataset (Sentiment140, revisión fija) | ✅ Cargado y verificado: 1.360.000 train / 240.000 test, coincide exacto con la guía |
| Muestra estratificada (200.000, semilla 42) + 3 folds | ✅ Generados en `protocol/partitions.csv` (folds balanceados 66.667/66.667/66.666) |
| `protocol/members.csv` | ✅ Generado con el mapeo de `MIEMBROS.md` |
| Experimento `nlp-lab2-sentiment140` en MLflow | ✅ Creado |
| Código del pipeline (`pipeline/`: config, data, preprocessing, representation, classifier, mlflow_logging) | ✅ Escrito, probado localmente contra el dataset real |
| API FastAPI (`api/main.py`: `/api/v1/predict`, `/audit/*`, `/health`) | ✅ Escrita contra el contrato del Anexo A.5, aún sin desplegar |
| **Run de protocolo registrado en MLflow (con artefactos)** | ⚠️ **Bloqueado** — ver nota abajo |
| T0 / B0 / comparaciones obligatorias / ablación / modelo final | ❌ Pendiente (depende de que el bloqueo de abajo se resuelva) |
| `reports/error_analysis.csv` / `.md` | ❌ Pendiente |

### ⚠️ Bloqueos activos

1. **Cuenta de AWS Academy desactivada** (`AWS account deactivated at 2026-09-12T10:06:26-07:00`) — no es falta de presupuesto ($4.3 de $50 usado). **Causa identificada**: según la documentación oficial de Vocareum, es el código de estado `SUSPENDED9` — *"Account has exceeded Sagemaker limit (fraud)"* — un guardrail de concurrencia de SageMaker que salta al crear varias Notebook Instances en poco tiempo (creamos las 5 en menos de 2 minutos). Reportado al profesor con esta explicación para que reactive la cuenta y, de paso, confirme el límite de concurrencia configurado. Los servicios ya desplegados (MLflow, este mismo repo) siguen respondiendo por HTTP mientras la cuenta esté así, pero no se pueden crear/gestionar recursos nuevos por CLI/consola.
2. **El servidor MLflow no tiene `--serve-artifacts` habilitado**: al intentar registrar el run de protocolo real, el registro de tags/params/métricas funcionó, pero la subida de artefactos (`protocol/partitions.csv`, `protocol/members.csv`) falló con `PermissionError: /opt/mlflow` — el cliente intenta escribir directo en una ruta que solo existe en el servidor. El run fallido se borró para no dejar basura. **Hay que reiniciar el servidor MLflow con `--serve-artifacts`** (o un artifact store remoto tipo S3) antes de poder registrar cualquier run con artefactos — esto incluye el run de protocolo y todos los runs experimentales/finales, que exigen artefactos obligatorios (Anexo A.2).

**Por precaución, cuando se reactive la cuenta:** crear/usar las Notebook Instances de a una, espaciadas en el tiempo (no las 5 de golpe) para no volver a disparar el guardrail. Ver [`PLAN.md`](PLAN.md).

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
