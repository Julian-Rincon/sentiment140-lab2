# Mapeo de integrantes — Laboratorio II (Análisis de Sentimientos)

El profesor no asignó `member_id` por integrante para esta actividad. El equipo definió el siguiente mapeo, que
se usa tal cual en `protocol/members.csv` (artefacto del run de protocolo en MLflow) y en el tag
`lab_member_id`/`notebook_arn` de cada run.

Cada integrante ejecuta sus runs experimentales desde **su propia cuenta individual de AWS Academy**, con **una**
Notebook Instance (rol `LabRole`). El archivo `/opt/ml/metadata/resource-metadata.json` de esa instancia se
registra sin editar como `provenance/sagemaker-resource-metadata.json` en cada run (Anexo A.2 de la guía).

| member_id | Integrante | notebook_arn |
|---|---|---|
| E01 | Julián Rincón | `arn:aws:sagemaker:us-east-1:170100747321:notebook-instance/nlp-lab2-e01` |
| E02 | Andrés Castro | *pendiente — se completa cuando cree su Notebook Instance* |
| E03 | Juan Hurtado | *pendiente — se completa cuando cree su Notebook Instance* |
| E04 | Miguel Flechas | *pendiente — se completa cuando cree su Notebook Instance* |
| E05 | Paula Caballero | *pendiente — se completa cuando cree su Notebook Instance* |
