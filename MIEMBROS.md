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
| E02 | Andrés Castro | `arn:aws:sagemaker:us-east-1:010843061983:notebook-instance/nlp-lab2-e02` |
| E03 | Juan Hurtado | `arn:aws:sagemaker:us-east-1:284743254744:notebook-instance/nlp-lab2-e03` |
| E04 | Miguel Flechas | `arn:aws:sagemaker:us-east-1:679786170402:notebook-instance/nlp-lab2-e04` |
| E05 | Paula Caballero | `arn:aws:sagemaker:us-east-1:473323351650:notebook-instance/nlp-lab2-e05` |
