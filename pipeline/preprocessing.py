"""
Transformaciones de preprocesamiento — sección 3 / A.3. Cada función recibe
un iterable de textos crudos y el dict `preprocessing` de una configuración
(o None para T0) y devuelve la lista de textos transformados.
"""
import re

import emoji as emoji_lib
import spacy

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_ELONGATION_RE = re.compile(r"(.)\1{2,}")  # 3+ repeticiones -> normaliza a 2 (spec: reduce_repeated_chars_to_2)

_nlp_cache = {}


def _get_spacy(model_name):
    if model_name not in _nlp_cache:
        _nlp_cache[model_name] = spacy.load(model_name, exclude=["parser", "ner"])
    return _nlp_cache[model_name]


def _apply_url_mention(text, url_mode, mention_mode):
    if url_mode == "drop":
        text = _URL_RE.sub("", text)
    elif url_mode.startswith("token:"):
        text = _URL_RE.sub(url_mode.split(":", 1)[1], text)
    if mention_mode == "drop":
        text = _MENTION_RE.sub("", text)
    elif mention_mode.startswith("token:"):
        text = _MENTION_RE.sub(mention_mode.split(":", 1)[1], text)
    return text


def preprocess_texts(texts, preprocessing_config):
    """preprocessing_config=None (caso T0) -> pass-through sin transformar."""
    if preprocessing_config is None:
        return list(texts)

    cfg = preprocessing_config
    out = list(texts)

    if cfg.get("lowercase"):
        out = [t.lower() for t in out]

    out = [_apply_url_mention(t, cfg.get("url", "keep"), cfg.get("mention", "keep")) for t in out]

    if cfg.get("emoji") == "text":
        out = [emoji_lib.demojize(t, delimiters=(" :", ": ")) for t in out]

    if cfg.get("elongation") == "normalize":
        out = [_ELONGATION_RE.sub(r"\1\1", t) for t in out]

    if cfg.get("whitespace") == "normalize":
        out = [_WHITESPACE_RE.sub(" ", t).strip() for t in out]

    stopwords_mode = cfg.get("stopwords", "keep")
    lemmatize = cfg.get("lemmatize", False)
    if stopwords_mode != "keep" or lemmatize:
        model = (cfg.get("resources") or {}).get("spacy_model", "en_core_web_md")
        nlp = _get_spacy(model)
        negators = set(cfg.get("negators") or [])
        processed = []
        for doc in nlp.pipe(out, batch_size=256):
            tokens = []
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue
                is_stop = tok.is_stop and tok.text.lower() not in negators
                if stopwords_mode in ("remove", "remove_preserve_negation") and is_stop:
                    continue
                tokens.append(tok.lemma_.lower() if lemmatize else tok.text)
            processed.append(" ".join(tokens))
        out = processed

    return out
