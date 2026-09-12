"""Clasificadores — sección 3 / Anexo A.3. classifier.type controlado."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

_BUILDERS = {
    # OJO: la guía exige que T0 desempate a "negative" (label 0). DummyClassifier
    # no garantiza esa regla de desempate por sí solo — verificar con los folds
    # reales; si hay empate y no cae en negative, hay que forzarlo manualmente.
    "most_frequent": lambda params: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda params: LogisticRegression(**params),
    "linear_svm": lambda params: LinearSVC(**params),
    "sgd": lambda params: SGDClassifier(**params),
}


def build_classifier(classifier_config):
    ctype = classifier_config["type"]
    if ctype not in _BUILDERS:
        raise ValueError(f"classifier.type desconocido: {ctype}")
    return _BUILDERS[ctype](classifier_config.get("parameters") or {})
