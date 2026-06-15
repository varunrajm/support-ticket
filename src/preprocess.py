"""
preprocess.py
Text cleaning and preprocessing utilities for the support ticket classifier.
"""
import re
import nltk
import pandas as pd


# ── NLTK data download ────────────────────────────────────────────────────────

def download_nltk_data() -> None:
    """Download required NLTK corpora (safe to call multiple times)."""
    resources = [
        ("tokenizers/punkt",         "punkt"),
        ("tokenizers/punkt_tab",     "punkt_tab"),
        ("corpora/stopwords",        "stopwords"),
        ("corpora/wordnet",          "wordnet"),
        ("corpora/omw-1.4",          "omw-1.4"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


# ── Core cleaning function ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove email addresses
      4. Remove HTML tags
      5. Remove non-alphabetic characters
      6. Normalize whitespace
      7. Remove stopwords + lemmatize
    """
    text = str(text).lower()

    # Remove noise
    text = re.sub(r"http\S+|www\.\S+", " ", text)     # URLs
    text = re.sub(r"\S+@\S+",           " ", text)     # emails
    text = re.sub(r"<[^>]+>",           " ", text)     # HTML
    text = re.sub(r"[^a-zA-Z\s]",       " ", text)     # non-alpha
    text = re.sub(r"\s+",               " ", text).strip()

    # Stopwords + lemmatization
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        tokens = [
            lemmatizer.lemmatize(w)
            for w in text.split()
            if w not in stop_words and len(w) > 2
        ]
    except Exception:
        # Graceful fallback if NLTK data is unavailable
        tokens = [w for w in text.split() if len(w) > 2]

    return " ".join(tokens)


# ── DataFrame-level preprocessing ────────────────────────────────────────────

def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "Ticket_Description",
    subject_column: str = "Ticket_Subject",
) -> pd.DataFrame:
    """
    Apply the full cleaning pipeline to a DataFrame and return the result
    with a new 'cleaned_text' column (description) and optionally a
    'cleaned_subject' column.
    """
    df_clean = df.copy()
    print(f"  Cleaning '{text_column}' column…")
    df_clean["cleaned_text"] = df_clean[text_column].astype(str).apply(clean_text)

    if subject_column and subject_column in df_clean.columns:
        print(f"  Cleaning '{subject_column}' column…")
        df_clean["cleaned_subject"] = df_clean[subject_column].astype(str).apply(clean_text)

    return df_clean


# ── Utility stats ─────────────────────────────────────────────────────────────

def get_text_stats(df: pd.DataFrame, text_column: str = "Ticket_Description") -> dict:
    """Return basic text length statistics for a given column."""
    lengths = df[text_column].astype(str).apply(len)
    word_counts = df[text_column].astype(str).apply(lambda x: len(x.split()))
    return {
        "mean_char_len":  lengths.mean(),
        "median_char_len": lengths.median(),
        "mean_words":     word_counts.mean(),
        "median_words":   word_counts.median(),
    }
