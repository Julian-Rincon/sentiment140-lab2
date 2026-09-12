"""
Representación del texto — sección 3 / Anexo A.4. Dado `representation_config`
y los textos ya preprocesados, devuelve (fit_fn, transform_fn) o un vectorizer
ya ajustado, según el tipo.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .preprocessing import _get_spacy


def build_vectorizer(representation_config):
    """bow / tfidf -> sklearn Vectorizer sin ajustar (se hace fit en train de cada fold)."""
    rep_type = representation_config["type"]
    ngram_range = tuple(representation_config["ngram_range"])
    if rep_type == "bow":
        return CountVectorizer(ngram_range=ngram_range)
    if rep_type == "tfidf":
        return TfidfVectorizer(ngram_range=ngram_range)
    raise ValueError(f"build_vectorizer no aplica a type={rep_type}, usar spacy_document_vectors")


def spacy_document_vectors(texts, representation_config):
    """
    R_SPACY: un vector por documento usando el modelo spaCy con vectores
    preentrenados. document_vector_method (parameters.document_vector_method)
    define cómo se agregan los vectores de token en uno solo por documento.
    """
    model = representation_config["spacy_model"]
    method = representation_config["parameters"]["document_vector_method"]
    nlp = _get_spacy(model)

    vectors = []
    for doc in nlp.pipe(texts, batch_size=256):
        if method == "mean_token_vectors":
            toks = [t.vector for t in doc if t.has_vector and not t.is_stop and not t.is_punct]
            vectors.append(np.mean(toks, axis=0) if toks else np.zeros(nlp.vocab.vectors_length))
        else:
            raise ValueError(f"document_vector_method desconocido: {method}")
    return np.vstack(vectors)
