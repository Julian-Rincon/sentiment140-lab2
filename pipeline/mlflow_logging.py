"""
Helpers de registro en MLflow — nombres exactos de tags/params/métricas/
artefactos del Anexo A.2. Un solo lugar para no desalinearse del contrato.
"""
import json
import os
import tempfile

import mlflow

from .data import DATASET_ID, DATASET_REVISION, SAMPLE_SIZE, RANDOM_SEED, CV_FOLDS

EXPERIMENT_NAME = "nlp-lab2-sentiment140"


def ensure_experiment():
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_protocol_run(partitions_csv_path, members_csv_path):
    """Run único con lab_run_type=protocol (A.2). No entrena ningún modelo."""
    ensure_experiment()
    with mlflow.start_run(run_name="protocol") as run:
        mlflow.set_tag("lab_run_type", "protocol")
        mlflow.log_param("dataset_id", DATASET_ID)
        mlflow.log_param("dataset_revision", DATASET_REVISION)
        mlflow.log_param("sampling_strategy", "stratified")
        mlflow.log_param("sample_size", SAMPLE_SIZE)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("cv_strategy", "StratifiedKFold")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("cv_shuffle", True)
        mlflow.log_artifact(partitions_csv_path, artifact_path="protocol")
        mlflow.log_artifact(members_csv_path, artifact_path="protocol")
        return run.info.run_id


def log_experimental_run(
    run_name, experiment_id_tag, stage, member_id, notebook_arn, configuration_id,
    configuration, protocol_run_id, fold_macro_f1, provenance_metadata_path,
    ablation_parent_run_id=None, ablation_reverted_decision=None,
):
    """
    Un run experimental (T0, B0, P_*, R_*, C_*, o ABLATION). fold_macro_f1 es
    una lista de 3 floats (uno por fold, en orden).
    """
    import statistics

    ensure_experiment()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "lab_run_type": "experiment",
            "lab_protocol_run_id": protocol_run_id,
            "lab_experiment_id": experiment_id_tag,
            "lab_stage": stage,
            "lab_member_id": member_id,
            "lab_configuration_id": configuration_id,
            "notebook_arn": notebook_arn,
        })
        if ablation_parent_run_id:
            mlflow.set_tag("lab_ablation_parent_run_id", ablation_parent_run_id)
            mlflow.log_param("ablation_reverted_decision", ablation_reverted_decision)

        mean_f1 = statistics.mean(fold_macro_f1)
        std_f1 = statistics.pstdev(fold_macro_f1)  # poblacional, ddof=0
        for i, f1 in enumerate(fold_macro_f1):
            mlflow.log_metric(f"macro_f1_fold_{i}", f1)
        mlflow.log_metric("macro_f1_mean", mean_f1)
        mlflow.log_metric("macro_f1_std", std_f1)

        if ablation_parent_run_id:
            client = mlflow.MlflowClient()
            parent = client.get_run(ablation_parent_run_id)
            parent_mean = parent.data.metrics["macro_f1_mean"]
            mlflow.log_metric("macro_f1_delta", parent_mean - mean_f1)

        _log_json_artifact(configuration, "run/configuration.json")
        mlflow.log_artifact(provenance_metadata_path, artifact_path="provenance")
        return run.info.run_id, mean_f1, std_f1


def log_final_run(
    member_id, notebook_arn, configuration_id, configuration, protocol_run_id,
    selected_experiment_run_id, test_macro_f1, provenance_metadata_path,
    error_analysis_csv_path, error_analysis_md_path,
):
    ensure_experiment()
    with mlflow.start_run(run_name="final") as run:
        mlflow.set_tags({
            "lab_run_type": "final",
            "lab_protocol_run_id": protocol_run_id,
            "lab_selected_experiment_run_id": selected_experiment_run_id,
            "lab_configuration_id": configuration_id,
            "lab_member_id": member_id,
            "notebook_arn": notebook_arn,
        })
        mlflow.log_param("training_size", 1_360_000)
        mlflow.log_metric("test_macro_f1", test_macro_f1)
        _log_json_artifact(configuration, "run/configuration.json")
        mlflow.log_artifact(provenance_metadata_path, artifact_path="provenance")
        mlflow.log_artifact(error_analysis_csv_path, artifact_path="reports")
        mlflow.log_artifact(error_analysis_md_path, artifact_path="reports")
        return run.info.run_id


def _log_json_artifact(obj, artifact_path):
    """artifact_path, ej. 'run/configuration.json' -> sube con ESE nombre exacto
    (el basename del archivo local debe coincidir, MLflow no permite renombrar)."""
    directory, filename = artifact_path.rsplit("/", 1)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    try:
        mlflow.log_artifact(tmp_path, artifact_path=directory)
    finally:
        os.unlink(tmp_path)
        os.rmdir(tmp_dir)
