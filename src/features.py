"""
features.py
Feature engineering utilities for the support ticket classifier.
Provides TF-IDF, Bag-of-Words, label encoding, and metadata features.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder


# ── Text feature extractors ───────────────────────────────────────────────────

def create_tfidf_features(
    texts,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
):
    """
    Fit a TF-IDF vectorizer on *texts* and return the sparse matrix plus
    the fitted vectorizer (so it can be used to transform test data later).

    Returns
    -------
    tfidf_matrix : scipy sparse matrix, shape (n_samples, max_features)
    vectorizer   : fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=2,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def create_bow_features(texts, max_features: int = 5000):
    """
    Fit a Bag-of-Words (CountVectorizer) on *texts*.

    Returns
    -------
    bow_matrix  : scipy sparse matrix
    vectorizer  : fitted CountVectorizer
    """
    vectorizer = CountVectorizer(max_features=max_features, min_df=2)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


# ── Label encoding ────────────────────────────────────────────────────────────

def encode_labels(y):
    """
    Encode a categorical array to integers.

    Returns
    -------
    y_encoded : np.ndarray of int
    encoder   : fitted LabelEncoder (needed to inverse_transform predictions)
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


# ── Feature importance helper ─────────────────────────────────────────────────

def get_top_features_per_class(vec, clf, class_names, n_top: int = 10) -> dict:
    """
    Extract the top *n_top* TF-IDF features (by logistic-regression coefficient)
    for each class.

    Parameters
    ----------
    vec          : fitted TfidfVectorizer (must expose get_feature_names_out)
    clf          : fitted classifier with .coef_ attribute (LogReg / LinearSVC)
    class_names  : list of class label strings
    n_top        : how many top terms to return per class

    Returns
    -------
    dict mapping class_name → [(term, score), …]
    """
    if not hasattr(clf, "coef_"):
        return {}

    feature_names = np.array(vec.get_feature_names_out())
    top_features: dict = {}

    for i, class_name in enumerate(class_names):
        if clf.coef_.shape[0] == 1:          # Binary classifier
            coefs = clf.coef_[0]
        else:                                 # Multi-class (OvR)
            coefs = clf.coef_[i]
        top_indices = coefs.argsort()[-n_top:][::-1]
        top_features[class_name] = [
            (feature_names[j], float(coefs[j])) for j in top_indices
        ]

    return top_features


# ── Metadata feature engineering ─────────────────────────────────────────────

URGENT_KEYWORDS = {
    "urgent", "asap", "critical", "immediately", "emergency",
    "locked out", "cannot access", "blocked", "security breach",
    "data loss", "down", "not working", "broken", "crashed",
}


def create_metadata_features(df: pd.DataFrame, text_col: str = "Ticket_Description") -> pd.DataFrame:
    """
    Engineer numerical metadata features from the raw ticket text.

    The returned DataFrame contains ONLY the new feature columns (not the
    original DataFrame columns) so it can be stacked cleanly in a
    ColumnTransformer.

    Features created
    ----------------
    text_length         : total character length of the description
    word_count          : number of whitespace-separated tokens
    exclamation_count   : number of '!' characters
    question_count      : number of '?' characters
    caps_word_ratio     : fraction of words that are fully capitalised (≥2 chars)
    avg_word_length     : mean characters per word
    urgent_keyword_count: count of high-priority signal words present
    """
    col = df[text_col].astype(str) if text_col in df.columns else pd.Series([""] * len(df))

    def count_urgent(text: str) -> int:
        text_lower = text.lower()
        return sum(1 for kw in URGENT_KEYWORDS if kw in text_lower)

    meta = pd.DataFrame(index=df.index)
    meta["text_length"]          = col.apply(len)
    meta["word_count"]           = col.apply(lambda x: len(x.split()))
    meta["exclamation_count"]    = col.apply(lambda x: x.count("!"))
    meta["question_count"]       = col.apply(lambda x: x.count("?"))
    meta["caps_word_ratio"]      = col.apply(
        lambda x: sum(1 for w in x.split() if len(w) >= 2 and w.isupper()) / max(len(x.split()), 1)
    )
    meta["avg_word_length"]      = col.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    meta["urgent_keyword_count"] = col.apply(count_urgent)

    return meta
