"""
Carga del dataset, muestra estratificada y folds — sección 2 / A.2 de la guía.
Determinista (semilla 42): correr esto en cualquier máquina produce la misma
muestra y los mismos folds, siempre que se use la misma revisión del dataset.
"""
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

DATASET_ID = "adilbekovich/Sentiment140Twitter"
DATASET_REVISION = "b6037e127257d95b9b23d31f78b264b9ebe697fd"
SAMPLE_SIZE = 200_000
RANDOM_SEED = 42
CV_FOLDS = 3


def load_train_test():
    """Carga train/test fijando la revisión exacta que exige la guía."""
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    return ds["train"], ds["test"]


def build_sample_and_folds(train_split):
    """
    Muestra estratificada de SAMPLE_SIZE registros de `train` (semilla 42),
    conservando el índice original de train (posición 0-based en el split).
    Luego 3 folds estratificados, mezcla aleatoria, semilla 42.

    Devuelve un DataFrame con columnas: index (posición original en train),
    label, text, fold (0/1/2 = en qué fold ese registro actúa como validación).
    """
    df = train_split.to_pandas().reset_index().rename(columns={"index": "index"})
    # `index` ya es la posición 0-based original en el split train (train_split.to_pandas()
    # conserva el orden nativo del split; reset_index() antes de cualquier muestreo/reindexado
    # asegura que se preserve, tal como exige la guía).

    sample_df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["label"] if "label" in df.columns else df["sentiment"],
        random_state=RANDOM_SEED,
    )
    sample_df = sample_df.sort_values("index").reset_index(drop=True)

    label_col = "label" if "label" in sample_df.columns else "sentiment"
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    sample_df["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(skf.split(sample_df, sample_df[label_col])):
        sample_df.loc[sample_df.index[val_idx], "fold"] = fold_id

    assert (sample_df["fold"] >= 0).all(), "quedaron filas sin fold asignado"
    return sample_df


def write_partitions_csv(sample_df, path):
    """protocol/partitions.csv — encabezado exacto index,fold, orden ascendente por index."""
    out = sample_df[["index", "fold"]].sort_values("index")
    out.to_csv(path, index=False, columns=["index", "fold"], encoding="utf-8")


def write_members_csv(members, path):
    """
    protocol/members.csv — encabezado exacto member_id,notebook_arn,
    una fila por integrante, orden lexicográfico ascendente por member_id.
    `members`: dict {member_id: notebook_arn}
    """
    rows = sorted(members.items())
    pd.DataFrame(rows, columns=["member_id", "notebook_arn"]).to_csv(path, index=False, encoding="utf-8")
