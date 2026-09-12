"""
Esquema de configuración efectiva (Anexo A.3 de la guía) y builders para cada
comparación obligatoria de la sección 3 / Anexo A.4. Cada builder devuelve un
dict serializable tal cual a `run/configuration.json`.

Las decisiones marcadas "DECISIÓN DEL EQUIPO" son las que la guía deja
explícitamente a criterio del equipo (negadores, specs de alargamiento/emoji,
modelo spaCy, método de vector por documento) — los valores aquí son una
propuesta razonable, no un requisito de la guía.
"""

SKLEARN_VERSION = "1.5.2"
SPACY_VERSION = "3.8.2"
SPACY_MODEL = "en_core_web_md"  # DECISIÓN DEL EQUIPO: tiene vectores preentrenados (necesario para R_SPACY)
SPACY_MODEL_VERSION = "3.8.0"

# DECISIÓN DEL EQUIPO: negadores preservados en stopwords=remove_preserve_negation.
NEGATORS_EN = ["not", "no", "never", "n't", "cannot", "cant", "nobody", "nothing", "neither", "nor"]


def _base_preprocessing(**overrides):
    base = {
        "lowercase": True,
        "url": "token:url",
        "mention": "token:user",
        "whitespace": "normalize",
        "stopwords": "keep",
        "negators": [],
        "lemmatize": False,
        "elongation": "keep",
        "elongation_spec": None,
        "emoji": "keep",
        "emoji_spec": None,
        "resources": {},
        "additional": {},
    }
    base.update(overrides)
    return base


def config_t0():
    return {
        "preprocessing": None,
        "representation": None,
        "classifier": {
            "type": "most_frequent",
            "library": "scikit-learn",
            "library_version": SKLEARN_VERSION,
            "parameters": {},
        },
    }


def config_b0():
    return {
        "preprocessing": _base_preprocessing(),
        "representation": {
            "type": "bow",
            "ngram_range": [1, 1],
            "library": "scikit-learn",
            "library_version": SKLEARN_VERSION,
            "spacy_model": None,
            "spacy_model_version": None,
            "parameters": {},
        },
        "classifier": {
            "type": "logistic_regression",
            "library": "scikit-learn",
            "library_version": SKLEARN_VERSION,
            "parameters": {},
        },
    }


# ---------------------------------------------------------------------------
# Etapa 1: preprocesamiento — cada una es B0 con UN campo distinto (A.4)
# ---------------------------------------------------------------------------

def config_p_stopwords():
    cfg = config_b0()
    cfg["preprocessing"]["stopwords"] = "remove"
    return cfg


def config_p_stopwords_negation():
    cfg = config_b0()
    cfg["preprocessing"]["stopwords"] = "remove_preserve_negation"
    cfg["preprocessing"]["negators"] = list(NEGATORS_EN)
    return cfg


def config_p_lemma():
    cfg = config_b0()
    cfg["preprocessing"]["lemmatize"] = True
    cfg["preprocessing"]["resources"] = {"spacy_model": SPACY_MODEL, "spacy_model_version": SPACY_MODEL_VERSION}
    return cfg


def config_p_elongation():
    cfg = config_b0()
    cfg["preprocessing"]["elongation"] = "normalize"
    cfg["preprocessing"]["elongation_spec"] = "reduce_repeated_chars_to_2"  # DECISIÓN DEL EQUIPO
    return cfg


def config_p_emoji():
    cfg = config_b0()
    cfg["preprocessing"]["emoji"] = "text"
    cfg["preprocessing"]["emoji_spec"] = "emoji.demojize"  # DECISIÓN DEL EQUIPO
    return cfg


# ---------------------------------------------------------------------------
# Etapa 2: representación — usa el preprocesamiento seleccionado (parámetro)
# ---------------------------------------------------------------------------

def config_r_bow(selected_preprocessing):
    return {
        "preprocessing": selected_preprocessing,
        "representation": {
            "type": "bow", "ngram_range": [1, 1],
            "library": "scikit-learn", "library_version": SKLEARN_VERSION,
            "spacy_model": None, "spacy_model_version": None, "parameters": {},
        },
        "classifier": {"type": "logistic_regression", "library": "scikit-learn", "library_version": SKLEARN_VERSION, "parameters": {}},
    }


def config_r_tfidf_uni(selected_preprocessing):
    cfg = config_r_bow(selected_preprocessing)
    cfg["representation"]["type"] = "tfidf"
    return cfg


def config_r_tfidf_uni_bi(selected_preprocessing):
    cfg = config_r_tfidf_uni(selected_preprocessing)
    cfg["representation"]["ngram_range"] = [1, 2]
    return cfg


def config_r_spacy(selected_preprocessing):
    return {
        "preprocessing": selected_preprocessing,
        "representation": {
            "type": "spacy_embedding", "ngram_range": [],
            "library": "spacy", "library_version": SPACY_VERSION,
            "spacy_model": SPACY_MODEL, "spacy_model_version": SPACY_MODEL_VERSION,
            "parameters": {"document_vector_method": "mean_token_vectors"},  # DECISIÓN DEL EQUIPO
        },
        "classifier": {"type": "logistic_regression", "library": "scikit-learn", "library_version": SKLEARN_VERSION, "parameters": {}},
    }


# ---------------------------------------------------------------------------
# Etapa 3: clasificador — usa preprocesamiento + representación seleccionados
# ---------------------------------------------------------------------------

def config_classifier(selected_preprocessing, selected_representation, classifier_type, parameters=None):
    return {
        "preprocessing": selected_preprocessing,
        "representation": selected_representation,
        "classifier": {
            "type": classifier_type,
            "library": "scikit-learn",
            "library_version": SKLEARN_VERSION,
            "parameters": parameters or {},
        },
    }


# ---------------------------------------------------------------------------
# Ablación (sección 4 / A.4) — revierte UNA decisión del candidato a B0
# ---------------------------------------------------------------------------

ABLATABLE_DECISIONS = {
    "preprocessing.stopwords",
    "preprocessing.lemmatize",
    "preprocessing.elongation",
    "preprocessing.emoji",
    "representation",
    "classifier",
}


def revert_decision(candidate_config, decision):
    """Devuelve una copia de candidate_config con `decision` revertida a B0."""
    import copy
    cfg = copy.deepcopy(candidate_config)
    b0 = config_b0()
    if decision not in ABLATABLE_DECISIONS:
        raise ValueError(f"decisión no ablacionable: {decision}")
    if decision == "preprocessing.stopwords":
        cfg["preprocessing"]["stopwords"] = "keep"
        cfg["preprocessing"]["negators"] = []
    elif decision == "preprocessing.lemmatize":
        cfg["preprocessing"]["lemmatize"] = False
    elif decision == "preprocessing.elongation":
        cfg["preprocessing"]["elongation"] = "keep"
        cfg["preprocessing"]["elongation_spec"] = None
    elif decision == "preprocessing.emoji":
        cfg["preprocessing"]["emoji"] = "keep"
        cfg["preprocessing"]["emoji_spec"] = None
    elif decision == "representation":
        cfg["representation"] = b0["representation"]
    elif decision == "classifier":
        cfg["classifier"] = b0["classifier"]
    return cfg
