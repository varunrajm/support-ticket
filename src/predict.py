"""
predict.py
Inference utilities for the support ticket auto-classification system.

Public API
----------
load_models()                              → (cat_pipeline, prio_pipeline, cat_enc, prio_enc)
classify_ticket(subject, desc, ...)        → dict with category, priority, confidence, routing
batch_classify(df, cat_model, prio_model)  → df with predictions appended
predict_interactive()                      → CLI loop for manual testing
"""
import os
import sys
import joblib
import pandas as pd

# Ensure project root is on the path regardless of invocation location
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocess import clean_text

# ── Model paths ───────────────────────────────────────────────────────────────
_MODELS_DIR       = os.path.join(_PROJECT_ROOT, "models")
_CAT_MODEL_PATH   = os.path.join(_MODELS_DIR, "category_classifier.pkl")
_CAT_ENC_PATH     = os.path.join(_MODELS_DIR, "category_label_encoder.pkl")
_PRIO_MODEL_PATH  = os.path.join(_MODELS_DIR, "priority_classifier.pkl")
_PRIO_ENC_PATH    = os.path.join(_MODELS_DIR, "priority_label_encoder.pkl")


# ── Routing rules ─────────────────────────────────────────────────────────────
_ROUTING_RULES = {
    ("Billing Issue",    "High"):   "ROUTE: Billing Escalation Queue  | SLA: 2 hours",
    ("Billing Issue",    "Medium"): "ROUTE: Billing Standard Queue    | SLA: 24 hours",
    ("Billing Issue",    "Low"):    "ROUTE: Billing Info Queue        | SLA: 72 hours",
    ("Technical Issue",  "High"):   "ROUTE: Tech Ops P1 Queue         | SLA: 1 hour",
    ("Technical Issue",  "Medium"): "ROUTE: Tech Ops Standard Queue   | SLA: 8 hours",
    ("Technical Issue",  "Low"):    "ROUTE: Tech Ops Low Queue        | SLA: 48 hours",
    ("Account Access",   "High"):   "ROUTE: Security/Auth Escalation  | SLA: 30 minutes",
    ("Account Access",   "Medium"): "ROUTE: Auth Support Queue        | SLA: 4 hours",
    ("Account Access",   "Low"):    "ROUTE: Auth Info Queue           | SLA: 48 hours",
    ("Product Inquiry",  "High"):   "ROUTE: Sales Priority Queue      | SLA: 2 hours",
    ("Product Inquiry",  "Medium"): "ROUTE: Sales Standard Queue      | SLA: 24 hours",
    ("Product Inquiry",  "Low"):    "ROUTE: Self-Service / FAQ        | SLA: 96 hours",
    ("Feature Request",  "High"):   "ROUTE: Product Management        | SLA: 48 hours",
    ("Feature Request",  "Medium"): "ROUTE: Product Backlog           | SLA: No SLA",
    ("Feature Request",  "Low"):    "ROUTE: Community Feedback        | SLA: No SLA",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models():
    """
    Load all four serialised artefacts from the models/ directory.

    Returns
    -------
    cat_model  : category classification pipeline
    prio_model : priority classification pipeline (expects a DataFrame with
                 'cleaned_text' + metadata columns)
    cat_enc    : LabelEncoder for categories
    prio_enc   : LabelEncoder for priorities
    """
    for path in [_CAT_MODEL_PATH, _CAT_ENC_PATH, _PRIO_MODEL_PATH, _PRIO_ENC_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run notebooks 04 and 05 first to train and save the models."
            )
    cat_model  = joblib.load(_CAT_MODEL_PATH)
    cat_enc    = joblib.load(_CAT_ENC_PATH)
    prio_model = joblib.load(_PRIO_MODEL_PATH)
    prio_enc   = joblib.load(_PRIO_ENC_PATH)
    return cat_model, prio_model, cat_enc, prio_enc


# ── Single-ticket inference ───────────────────────────────────────────────────

def _get_confidence(pipeline, text_input, label_enc) -> tuple[str, float]:
    """
    Run *pipeline* on *text_input* and return (label, confidence_0_to_1).
    Works with pipelines that expose predict_proba OR decision_function.
    *text_input* may be a string (for text-only pipelines) or a DataFrame
    row (for hybrid pipelines).
    """
    # Wrap scalar strings so sklearn gets an iterable
    if isinstance(text_input, str):
        X = [text_input]
    elif isinstance(text_input, pd.Series):
        X = text_input.to_frame().T
    else:
        X = text_input  # already a DataFrame

    if hasattr(pipeline, "predict_proba"):
        probs   = pipeline.predict_proba(X)[0]
        idx     = int(probs.argmax())
        conf    = float(probs[idx])
    else:
        # LinearSVC uses decision_function — use softmax-like scaling
        scores  = pipeline.decision_function(X)[0]
        scores  = scores - scores.max()
        import numpy as np
        exp     = np.exp(scores)
        probs   = exp / exp.sum()
        idx     = int(probs.argmax())
        conf    = float(probs[idx])

    label = label_enc.inverse_transform([idx])[0]
    return label, conf


def classify_ticket(
    subject: str,
    description: str,
    cat_model,
    prio_model,
    cat_enc=None,
    prio_enc=None,
) -> dict:
    """
    Classify a single support ticket.

    Parameters
    ----------
    subject      : raw ticket subject line
    description  : raw ticket body text
    cat_model    : fitted category pipeline
    prio_model   : fitted priority pipeline
    cat_enc      : LabelEncoder for categories  (loaded from disk if None)
    prio_enc     : LabelEncoder for priorities  (loaded from disk if None)

    Returns
    -------
    dict with keys:
        category, category_confidence,
        priority, priority_confidence,
        recommended_action
    """
    import numpy as np

    if cat_enc is None or prio_enc is None:
        _, _, cat_enc_loaded, prio_enc_loaded = load_models()
        cat_enc  = cat_enc  or cat_enc_loaded
        prio_enc = prio_enc or prio_enc_loaded

    # --- Category prediction (text-only pipeline) ---
    combined_text  = f"{subject} {description}"
    cleaned        = clean_text(combined_text)
    category, cat_conf = _get_confidence(cat_model, cleaned, cat_enc)

    # --- Priority prediction (hybrid pipeline: needs DataFrame) ---
    meta_cols = ["text_length", "word_count", "exclamation_count",
                 "question_count", "caps_word_ratio", "avg_word_length",
                 "urgent_keyword_count"]

    raw_text = combined_text
    row_data = {
        "Ticket_Description": raw_text,
        "cleaned_text":       cleaned,
        "text_length":        len(raw_text),
        "word_count":         len(raw_text.split()),
        "exclamation_count":  raw_text.count("!"),
        "question_count":     raw_text.count("?"),
        "caps_word_ratio":    sum(1 for w in raw_text.split() if len(w) >= 2 and w.isupper()) / max(len(raw_text.split()), 1),
        "avg_word_length":    float(np.mean([len(w) for w in raw_text.split()])) if raw_text.split() else 0.0,
        "urgent_keyword_count": sum(1 for kw in [
            "urgent", "asap", "critical", "immediately", "emergency",
            "locked out", "cannot access", "blocked", "down"
        ] if kw in raw_text.lower()),
    }
    row_df = pd.DataFrame([row_data])
    priority, prio_conf = _get_confidence(prio_model, row_df, prio_enc)

    routing = _ROUTING_RULES.get((category, priority), "ROUTE: General Support Queue")

    return {
        "category":             category,
        "category_confidence":  cat_conf,
        "priority":             priority,
        "priority_confidence":  prio_conf,
        "recommended_action":   routing,
    }


# ── Batch inference ───────────────────────────────────────────────────────────

def batch_classify(df: pd.DataFrame, cat_model, prio_model,
                   cat_enc=None, prio_enc=None) -> pd.DataFrame:
    """
    Classify a DataFrame of raw tickets.

    Expects columns: 'Ticket_Subject', 'Ticket_Description'
    Appends: 'Predicted_Category', 'Category_Confidence',
             'Predicted_Priority',  'Priority_Confidence',
             'Recommended_Routing_Action'
    """
    if cat_enc is None or prio_enc is None:
        _, _, cat_enc_loaded, prio_enc_loaded = load_models()
        cat_enc  = cat_enc  or cat_enc_loaded
        prio_enc = prio_enc or prio_enc_loaded

    results = df.apply(
        lambda row: classify_ticket(
            str(row.get("Ticket_Subject", "")),
            str(row.get("Ticket_Description", "")),
            cat_model, prio_model, cat_enc, prio_enc,
        ),
        axis=1,
    )

    df_out = df.copy()
    df_out["Predicted_Category"]         = results.apply(lambda r: r["category"])
    df_out["Category_Confidence"]        = results.apply(lambda r: f"{r['category_confidence']:.1%}")
    df_out["Predicted_Priority"]         = results.apply(lambda r: r["priority"])
    df_out["Priority_Confidence"]        = results.apply(lambda r: f"{r['priority_confidence']:.1%}")
    df_out["Recommended_Routing_Action"] = results.apply(lambda r: r["recommended_action"])
    return df_out


# ── Interactive CLI ───────────────────────────────────────────────────────────

def predict_interactive() -> None:
    """Run an interactive command-line classification session."""
    print("Loading models…")
    cat_model, prio_model, cat_enc, prio_enc = load_models()
    print("Models loaded.\n")
    print("=" * 65)
    print("  CUSTOMER SUPPORT TICKET AUTO-CLASSIFICATION SYSTEM")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 65)

    while True:
        print()
        subject = input("Ticket subject : ").strip()
        if subject.lower() in ("quit", "exit"):
            break
        description = input("Ticket body    : ").strip()
        if not description:
            continue

        result = classify_ticket(subject, description, cat_model, prio_model, cat_enc, prio_enc)

        print("\n── Prediction ──────────────────────────────────────────────")
        print(f"  Category : {result['category']}  ({result['category_confidence']:.1%} confidence)")
        print(f"  Priority : {result['priority']}  ({result['priority_confidence']:.1%} confidence)")
        print(f"  Action   : {result['recommended_action']}")
        print("─" * 65)


if __name__ == "__main__":
    predict_interactive()
