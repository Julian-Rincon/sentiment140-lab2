"""
API de Análisis de Sentimientos — implementa el contrato del Anexo A.5 de la
guía (Laboratorio II - 2026 S02). Reutiliza los patrones ya probados en la
Actividad 1: UTF8JSONResponse (evita mojibake en JSON), validación estricta
"todo o nada" vía Pydantic, middleware de logging, CORS abierto.

Los endpoints /audit/* consultan MLflow EN TIEMPO DE EJECUCIÓN (nunca una
copia local) — si MLflow no está disponible, devuelven 503 mlflow_unavailable
tal como exige el Anexo A.5.
"""
import logging
import time

import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, field_validator

MLFLOW_TRACKING_URI = "http://3.90.102.99:5000"
EXPERIMENT_NAME = "nlp-lab2-sentiment140"
MODEL_NAME = "sentiment140"
MODEL_ALIAS = "champion"
MAX_TEXT_LENGTH = 1_000
MAX_BATCH_SIZE = 32

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sentiment_api")


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


app = FastAPI(
    default_response_class=UTF8JSONResponse,
    title="Sentiment140 API",
    description="Contrato del Laboratorio II - 2026 S02 (Análisis de sentimientos)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - t0) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


def _client() -> MlflowClient:
    try:
        return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc


def _mlflow_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail="mlflow_unavailable") from exc


# ---------------------------------------------------------------------------
# /api/v1/predict
# ---------------------------------------------------------------------------

def _validate_text(v: str) -> str:
    if not isinstance(v, str):
        raise ValueError("text: cada elemento debe ser string")
    if len(v) > MAX_TEXT_LENGTH:
        raise ValueError(f"text: texto demasiado largo ({len(v)} chars), máximo {MAX_TEXT_LENGTH}")
    if not v.strip():
        raise ValueError("text: no puede estar vacío ni contener solo espacios")
    return v


class PredictIn(BaseModel):
    text: str | list[str]

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if isinstance(v, str):
            return [_validate_text(v)]
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("text: el lote no puede estar vacío")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"text: lote demasiado grande ({len(v)}), máximo {MAX_BATCH_SIZE}")
        return [_validate_text(item) for item in v]


def _resolve_champion_model():
    """sentiment140@champion -> (model_version, run_id). 404/409 según A.5."""
    client = _client()
    try:
        mv = _mlflow_call(client.get_model_version_by_alias, MODEL_NAME, MODEL_ALIAS)
    except HTTPException:
        raise
    except MlflowException as exc:
        raise HTTPException(status_code=404, detail="champion_not_found") from exc

    run_id = mv.run_id
    run = _mlflow_call(client.get_run, run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("lab_run_type") != "final":
        raise HTTPException(status_code=409, detail="champion_invalid")
    return mv, run


_PYFUNC_MODEL_CACHE = {}


def _load_pyfunc_model(run_id):
    if run_id not in _PYFUNC_MODEL_CACHE:
        _PYFUNC_MODEL_CACHE[run_id] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    return _PYFUNC_MODEL_CACHE[run_id]


@app.post("/api/v1/predict")
def predict(body: PredictIn):
    mv, run = _resolve_champion_model()
    model = _load_pyfunc_model(run.info.run_id)
    predictions = list(model.predict(body.text))
    return {"model_run_id": run.info.run_id, "predictions": predictions}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        mv, run = _resolve_champion_model()
        _load_pyfunc_model(run.info.run_id)
        return {"status": "ok", "model_run_id": run.info.run_id}
    except HTTPException:
        return JSONResponse(status_code=503, content={"status": "unavailable", "model_run_id": None})


# ---------------------------------------------------------------------------
# /audit/protocol
# ---------------------------------------------------------------------------

@app.get("/audit/protocol")
def audit_protocol():
    client = _client()
    experiment = _mlflow_call(client.get_experiment_by_name, EXPERIMENT_NAME)
    if experiment is None:
        raise HTTPException(status_code=409, detail="protocol_not_unique")
    runs = _mlflow_call(
        client.search_runs,
        [experiment.experiment_id],
        filter_string="tags.lab_run_type = 'protocol'",
    )
    if len(runs) != 1:
        raise HTTPException(status_code=409, detail="protocol_not_unique")
    run = runs[0]
    p = run.data.params
    return {
        "protocol_run_id": run.info.run_id,
        "dataset_id": p.get("dataset_id"),
        "dataset_revision": p.get("dataset_revision"),
        "sampling_strategy": p.get("sampling_strategy"),
        "sample_size": int(p.get("sample_size", 0)),
        "random_seed": int(p.get("random_seed", 0)),
        "cv_strategy": p.get("cv_strategy"),
        "cv_folds": int(p.get("cv_folds", 0)),
        "cv_shuffle": p.get("cv_shuffle", "").lower() == "true",
        "partitions_artifact": "protocol/partitions.csv",
        "members_artifact": "protocol/members.csv",
    }


# ---------------------------------------------------------------------------
# /audit/runs
# ---------------------------------------------------------------------------

def _list_run_artifacts(client, run_id, path=""):
    paths = []
    for f in _mlflow_call(client.list_artifacts, run_id, path):
        if f.is_dir:
            paths.extend(_list_run_artifacts(client, run_id, f.path))
        else:
            paths.append(f.path)
    return sorted(paths)


@app.get("/audit/runs")
def audit_runs():
    client = _client()
    experiment = _mlflow_call(client.get_experiment_by_name, EXPERIMENT_NAME)
    if experiment is None:
        return {"runs": []}
    all_runs = _mlflow_call(client.search_runs, [experiment.experiment_id], filter_string="")
    presented = [r for r in all_runs if r.data.tags.get("lab_run_type") in ("protocol", "experiment", "final")]

    out = []
    for r in presented:
        configuration = None
        run_type = r.data.tags.get("lab_run_type")
        if run_type in ("experiment", "final"):
            local_path = _mlflow_call(client.download_artifacts, r.info.run_id, "run/configuration.json") \
                if _artifact_exists(client, r.info.run_id, "run/configuration.json") else None
            if local_path:
                import json
                try:
                    with open(local_path) as fh:
                        configuration = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    configuration = None
        out.append({
            "run_id": r.info.run_id,
            "status": r.info.status,
            "run_type": run_type,
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
            "tags": dict(r.data.tags),
            "artifacts": _list_run_artifacts(client, r.info.run_id),
            "configuration": configuration,
        })
    out.sort(key=lambda x: x["run_id"])
    return {"runs": out}


def _artifact_exists(client, run_id, path):
    try:
        parent = "/".join(path.split("/")[:-1])
        names = {f.path for f in client.list_artifacts(run_id, parent)}
        return path in names
    except MlflowException:
        return False


# ---------------------------------------------------------------------------
# /audit/contributions
# ---------------------------------------------------------------------------

@app.get("/audit/contributions")
def audit_contributions():
    client = _client()
    experiment = _mlflow_call(client.get_experiment_by_name, EXPERIMENT_NAME)
    if experiment is None:
        return {"members": [], "invalid_run_ids": [], "unattributed_run_ids": []}

    all_runs = _mlflow_call(client.search_runs, [experiment.experiment_id], filter_string="")
    exp_runs = [r for r in all_runs if r.data.tags.get("lab_run_type") == "experiment"]

    by_member = {}
    invalid_run_ids = []
    unattributed_run_ids = []

    for r in exp_runs:
        member_id = r.data.tags.get("lab_member_id")
        notebook_arn = r.data.tags.get("notebook_arn")
        config_id = r.data.tags.get("lab_configuration_id")
        stage = r.data.tags.get("lab_stage")
        experiment_id_tag = r.data.tags.get("lab_experiment_id")

        if r.info.status != "FINISHED" or not member_id or not notebook_arn:
            invalid_run_ids.append(r.info.run_id)
            if not member_id or not notebook_arn:
                unattributed_run_ids.append(r.info.run_id)
            continue

        entry = by_member.setdefault(member_id, {
            "member_id": member_id, "notebook_arn": notebook_arn,
            "run_ids": [], "counted_run_ids": [], "configuration_ids": set(), "stages": set(),
        })
        entry["run_ids"].append(r.info.run_id)
        if experiment_id_tag not in ("T0", "B0") and config_id:
            entry["counted_run_ids"].append(r.info.run_id)
            entry["configuration_ids"].add(config_id)
            if stage:
                entry["stages"].add(stage)

    members = []
    for member_id in sorted(by_member):
        e = by_member[member_id]
        members.append({
            "member_id": e["member_id"],
            "notebook_arn": e["notebook_arn"],
            "run_ids": sorted(e["run_ids"]),
            "counted_run_ids": sorted(e["counted_run_ids"]),
            "configuration_ids": sorted(e["configuration_ids"]),
            "valid_configurations": len(e["configuration_ids"]),
            "stages": sorted(e["stages"]),
        })

    return {
        "members": members,
        "invalid_run_ids": sorted(set(invalid_run_ids)),
        "unattributed_run_ids": sorted(set(unattributed_run_ids)),
    }


# ---------------------------------------------------------------------------
# /audit/model
# ---------------------------------------------------------------------------

@app.get("/audit/model")
def audit_model():
    client = _client()
    mv, run = _resolve_champion_model()
    import json
    config = None
    try:
        local_path = client.download_artifacts(run.info.run_id, "run/configuration.json")
        with open(local_path) as fh:
            config = json.load(fh)
    except (MlflowException, OSError, json.JSONDecodeError):
        config = None

    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": int(mv.version),
        "run_id": run.info.run_id,
        "protocol_run_id": run.data.tags.get("lab_protocol_run_id"),
        "selected_experiment_run_id": run.data.tags.get("lab_selected_experiment_run_id"),
        "configuration_id": run.data.tags.get("lab_configuration_id"),
        "configuration": config,
        "training_size": int(run.data.params.get("training_size", 0)),
        "test_macro_f1": float(run.data.metrics.get("test_macro_f1", 0.0)),
    }
