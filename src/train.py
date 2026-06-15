"""
train.py
Model training, evaluation, hyperparameter tuning, and persistence utilities.
"""
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score


# ── Model catalogue ───────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a fresh dictionary of untrained classifiers."""
    return {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        "Linear SVC":          LinearSVC(random_state=42, dual="auto", max_iter=2000),
        "Random Forest":       RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
    }


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, models: dict) -> dict:
    """
    Fit each model in *models* on *(X_train, y_train)* and evaluate on
    *(X_test, y_test)*.

    Returns
    -------
    dict  model_name → {accuracy, f1_macro, f1_weighted, model}
    """
    results = {}
    for name, model in models.items():
        print(f"  Training {name}…", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1m  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        results[name] = {
            "accuracy":    acc,
            "f1_macro":    f1m,
            "f1_weighted": f1w,
            "model":       model,
        }
        print(f"Accuracy={acc:.3f}  Macro-F1={f1m:.3f}")
    return results


# ── Hyperparameter tuning: Category (text-only pipeline) ─────────────────────

def tune_best_model(X_train_raw, y_train, vectorizer=None) -> Pipeline:
    """
    Grid-search a TF-IDF + Logistic Regression pipeline on raw text.
    *vectorizer* is accepted but ignored — a fresh vectorizer is created
    inside the pipeline so GridSearchCV can tune its parameters too.

    Returns
    -------
    best_estimator : fitted sklearn Pipeline
    """
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(sublinear_tf=True, min_df=2, strip_accents="unicode")),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
    ])

    param_grid = {
        "vectorizer__max_features": [5000, 10000],
        "vectorizer__ngram_range":  [(1, 1), (1, 2)],
        "classifier__C":            [0.1, 1.0, 10.0],
    }

    print("  Running GridSearchCV for Category model (this may take a minute)…")
    gs = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=0)
    gs.fit(X_train_raw, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.3f}")
    return gs.best_estimator_


# ── Hyperparameter tuning: Priority (hybrid pipeline) ────────────────────────

def tune_priority_model(df_train, y_train, meta_cols: list) -> Pipeline:
    """
    Grid-search a hybrid TF-IDF + metadata + Logistic Regression pipeline.

    Parameters
    ----------
    df_train   : DataFrame with 'cleaned_text' column + meta_cols
    y_train    : encoded priority labels
    meta_cols  : list of numeric metadata column names

    Returns
    -------
    best_estimator : fitted sklearn Pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True), "cleaned_text"),
            ("meta", MinMaxScaler(), meta_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(random_state=42, max_iter=1000)),
    ])

    param_grid = {"classifier__C": [0.1, 1.0, 5.0, 10.0]}

    print("  Running GridSearchCV for Priority model (hybrid pipeline)…")
    gs = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(df_train, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.3f}")
    return gs.best_estimator_


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """Serialise *model* to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved → {path}")


def load_model(path: str):
    """Load and return a joblib-serialised model."""
    return joblib.load(path)
