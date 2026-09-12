# Mapeo de integrantes — Laboratorio II (Análisis de Sentimientos)

El profesor no asignó `member_id` ni Notebook Instance por integrante para esta actividad. El equipo definió el
siguiente mapeo, que se usará tal cual en `protocol/members.csv` (artefacto del run de protocolo en MLflow) y en
el tag `lab_member_id`/`notebook_arn` de cada run.

| member_id | Integrante | notebook_arn |
|---|---|---|
| E01 | Julián Rincón | arn:aws:sagemaker:us-east-1:548774546300:notebook-instance/nlp-lab2-e01 |
| E02 | Andrés Castro | arn:aws:sagemaker:us-east-1:548774546300:notebook-instance/nlp-lab2-e02 |
| E03 | Juan Hurtado | arn:aws:sagemaker:us-east-1:548774546300:notebook-instance/nlp-lab2-e03 |
| E04 | Miguel Flechas | arn:aws:sagemaker:us-east-1:548774546300:notebook-instance/nlp-lab2-e04 |
| E05 | Paula Caballero | arn:aws:sagemaker:us-east-1:548774546300:notebook-instance/nlp-lab2-e05 |

Cada Notebook Instance es un recurso SageMaker real (`ml.t3.medium`, rol `LabRole`), creado el 2026-09-12 en la
cuenta de AWS Academy del equipo. Cada integrante debe ejecutar sus runs experimentales desde **su propia**
instancia asignada — el archivo `/opt/ml/metadata/resource-metadata.json` de esa instancia se registra sin
editar como `provenance/sagemaker-resource-metadata.json` en cada run (Anexo A.2 de la guía).
