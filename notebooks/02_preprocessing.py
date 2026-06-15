# %% [markdown]
# # 🧹 02 — Text Preprocessing
# **Customer Support Ticket Classification Project**
#
# This notebook demonstrates the full text preprocessing pipeline applied to
# customer support ticket descriptions. We walk through each cleaning step,
# visualize the effects, and save the processed dataset.
#
# **Sections:**
# 1. Setup & Data Loading
# 2. Raw Text Examples
# 3. Step-by-step Cleaning Demo
# 4. Full Dataset Preprocessing
# 5. Before / After Comparison
# 6. Vocabulary & Token Statistics
# 7. Save Cleaned Data

# %%
# ── Setup ────────────────────────────────────────────────────────────────────
import sys
import os
import warnings

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('H:/future interns/2'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
PALETTE = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6',
           '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
sns.set_palette(PALETTE)

FIGDIR = os.path.abspath(os.path.join('..', 'reports', 'figures'))
os.makedirs(FIGDIR, exist_ok=True)

# Import project modules
from src.preprocess import download_nltk_data, clean_text, preprocess_dataframe, get_text_stats

# %%
# ── Download NLTK data ──────────────────────────────────────────────────────
download_nltk_data()

# %% [markdown]
# ---
# ## 1 · Load Raw Data

# %%
DATA_PATH = os.path.join('..', 'data', 'raw', 'customer_support_tickets.csv')
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Text column: 'Ticket_Description'")
print(f"Subject column: 'Ticket_Subject'")

# %% [markdown]
# ---
# ## 2 · Raw Text Examples
# Let's examine 5 raw ticket descriptions to identify the types of noise
# we need to handle: special characters, HTML tags, extra whitespace,
# mixed casing, punctuation, etc.

# %%
print("=" * 80)
print("RAW TICKET DESCRIPTIONS (5 examples)")
print("=" * 80)

sample_indices = df.sample(5, random_state=42).index
for i, idx in enumerate(sample_indices, 1):
    text = str(df.loc[idx, 'Ticket_Description'])
    subject = str(df.loc[idx, 'Ticket_Subject'])
    ticket_type = str(df.loc[idx, 'Ticket_Type'])
    print(f"\n{'─' * 80}")
    print(f"Example {i} | Type: {ticket_type} | Subject: {subject}")
    print(f"{'─' * 80}")
    print(f"  \"{text[:300]}{'...' if len(text) > 300 else ''}\"")
    print(f"  [Length: {len(text)} chars | Words: {len(text.split())}]")

# %% [markdown]
# ---
# ## 3 · Step-by-step Cleaning Demo
# Walk through the cleaning pipeline on a single example, showing the
# intermediate state after each transformation.

# %%
# Pick one representative example
demo_idx = sample_indices[0]
demo_text = str(df.loc[demo_idx, 'Ticket_Description'])
print("ORIGINAL TEXT:")
print(f"  \"{demo_text}\"\n")

# Step 1: Lowercase
step1 = demo_text.lower()
print("STEP 1 — Lowercase:")
print(f"  \"{step1}\"\n")

# Step 2: Remove URLs
step2 = re.sub(r'http\S+|www\.\S+', '', step1)
print("STEP 2 — Remove URLs:")
print(f"  \"{step2}\"\n")

# Step 3: Remove email addresses
step3 = re.sub(r'\S+@\S+', '', step2)
print("STEP 3 — Remove Emails:")
print(f"  \"{step3}\"\n")

# Step 4: Remove HTML tags
step4 = re.sub(r'<[^>]+>', '', step3)
print("STEP 4 — Remove HTML Tags:")
print(f"  \"{step4}\"\n")

# Step 5: Remove special characters and digits
step5 = re.sub(r'[^a-zA-Z\s]', '', step4)
print("STEP 5 — Remove Special Characters & Digits:")
print(f"  \"{step5}\"\n")

# Step 6: Remove extra whitespace
step6 = ' '.join(step5.split())
print("STEP 6 — Normalize Whitespace:")
print(f"  \"{step6}\"\n")

# Step 7: Full clean_text function
step7 = clean_text(demo_text)
print("FULL clean_text() OUTPUT:")
print(f"  \"{step7}\"")

# %% [markdown]
# ---
# ## 4 · Full Dataset Preprocessing
# Apply the complete preprocessing pipeline to the entire dataset using
# `preprocess_dataframe()`.

# %%
print("Applying preprocessing pipeline to full dataset...")
print("This may take a moment...\n")

df_clean = preprocess_dataframe(
    df.copy(),
    text_column='Ticket_Description',
    subject_column='Ticket_Subject'
)

print(f"✅ Preprocessing complete!")
print(f"   Original shape: {df.shape}")
print(f"   Cleaned shape:  {df_clean.shape}")
print(f"\nNew/modified columns:")
for col in df_clean.columns:
    if col not in df.columns:
        print(f"   + {col}")

# %% [markdown]
# ---
# ## 5 · Before / After Comparison
# Show side-by-side comparisons of original and cleaned text for 10
# randomly sampled tickets.

# %%
# Determine the cleaned text column name
if 'cleaned_text' in df_clean.columns:
    clean_col = 'cleaned_text'
elif 'clean_text' in df_clean.columns:
    clean_col = 'clean_text'
elif 'processed_text' in df_clean.columns:
    clean_col = 'processed_text'
else:
    # Fallback: apply clean_text manually
    df_clean['cleaned_text'] = df['Ticket_Description'].astype(str).apply(clean_text)
    clean_col = 'cleaned_text'

print("=" * 100)
print("BEFORE / AFTER COMPARISON (10 examples)")
print("=" * 100)

comparison_indices = df.sample(10, random_state=123).index
for i, idx in enumerate(comparison_indices, 1):
    original = str(df.loc[idx, 'Ticket_Description'])[:150]
    cleaned = str(df_clean.loc[idx, clean_col])[:150] if idx in df_clean.index else "N/A"
    print(f"\n{'─' * 100}")
    print(f"Example {i:2d}:")
    print(f"  BEFORE: \"{original}{'...' if len(str(df.loc[idx, 'Ticket_Description'])) > 150 else ''}\"")
    print(f"  AFTER:  \"{cleaned}{'...' if len(str(df_clean.loc[idx, clean_col])) > 150 else ''}\"")

# %%
# ── Formatted comparison table ──────────────────────────────────────────────
comparison_df = pd.DataFrame({
    'Original (first 100 chars)': df.loc[comparison_indices, 'Ticket_Description']
                                    .astype(str).str[:100],
    'Cleaned (first 100 chars)': df_clean.loc[comparison_indices, clean_col]
                                    .astype(str).str[:100],
    'Ticket_Type': df.loc[comparison_indices, 'Ticket_Type']
})
print("\n📋 Comparison Table:")
print(comparison_df.to_string(index=False))

# %% [markdown]
# ---
# ## 6 · Vocabulary & Token Statistics
# Compare vocabulary sizes, token count distributions, and other text
# statistics before and after preprocessing.

# %%
# ── Vocabulary Size ──────────────────────────────────────────────────────────
original_vocab = set()
for text in df['Ticket_Description'].astype(str):
    original_vocab.update(text.lower().split())

cleaned_vocab = set()
for text in df_clean[clean_col].astype(str):
    cleaned_vocab.update(text.split())

print("VOCABULARY COMPARISON")
print(f"  Before cleaning: {len(original_vocab):,} unique tokens")
print(f"  After cleaning:  {len(cleaned_vocab):,} unique tokens")
print(f"  Reduction:       {len(original_vocab) - len(cleaned_vocab):,} tokens "
      f"({(1 - len(cleaned_vocab)/max(len(original_vocab),1))*100:.1f}%)")

# %%
# ── Token Count Distributions (Overlaid Histograms) ─────────────────────────
original_lengths = df['Ticket_Description'].astype(str).apply(lambda x: len(x.split()))
cleaned_lengths = df_clean[clean_col].astype(str).apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram overlay
axes[0].hist(original_lengths, bins=40, alpha=0.6, color=PALETTE[2],
             label='Before', edgecolor='white')
axes[0].hist(cleaned_lengths, bins=40, alpha=0.6, color=PALETTE[0],
             label='After', edgecolor='white')
axes[0].axvline(original_lengths.mean(), color=PALETTE[2], linestyle='--', linewidth=2)
axes[0].axvline(cleaned_lengths.mean(), color=PALETTE[0], linestyle='--', linewidth=2)
axes[0].set_title('Token Count Distribution: Before vs After', fontsize=14,
                  fontweight='bold')
axes[0].set_xlabel('Number of Tokens', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].legend(fontsize=12)

# Character length comparison
original_chars = df['Ticket_Description'].astype(str).apply(len)
cleaned_chars = df_clean[clean_col].astype(str).apply(len)

axes[1].hist(original_chars, bins=40, alpha=0.6, color=PALETTE[2],
             label='Before', edgecolor='white')
axes[1].hist(cleaned_chars, bins=40, alpha=0.6, color=PALETTE[0],
             label='After', edgecolor='white')
axes[1].set_title('Character Length Distribution: Before vs After', fontsize=14,
                  fontweight='bold')
axes[1].set_xlabel('Character Length', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'preprocessing_comparison.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: preprocessing_comparison.png")

# %%
# ── Summary Statistics Table ─────────────────────────────────────────────────
stats_df = pd.DataFrame({
    'Metric': ['Mean token count', 'Median token count', 'Mean char length',
               'Median char length', 'Vocabulary size', 'Empty texts'],
    'Before': [
        f"{original_lengths.mean():.1f}",
        f"{original_lengths.median():.0f}",
        f"{original_chars.mean():.1f}",
        f"{original_chars.median():.0f}",
        f"{len(original_vocab):,}",
        f"{(original_lengths == 0).sum()}"
    ],
    'After': [
        f"{cleaned_lengths.mean():.1f}",
        f"{cleaned_lengths.median():.0f}",
        f"{cleaned_chars.mean():.1f}",
        f"{cleaned_chars.median():.0f}",
        f"{len(cleaned_vocab):,}",
        f"{(cleaned_lengths == 0).sum()}"
    ]
})
print("\n📊 Preprocessing Impact Summary:")
print(stats_df.to_string(index=False))

# %%
# ── Vocabulary Bar Chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['Before Cleaning', 'After Cleaning'],
              [len(original_vocab), len(cleaned_vocab)],
              color=[PALETTE[2], PALETTE[0]], edgecolor='white', width=0.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{int(bar.get_height()):,}', ha='center', fontweight='bold', fontsize=13)
ax.set_title('Vocabulary Size: Before vs After Cleaning', fontsize=15, fontweight='bold')
ax.set_ylabel('Unique Tokens', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'vocabulary_comparison.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: vocabulary_comparison.png")

# %% [markdown]
# ---
# ## 7 · Save Cleaned Data
# Persist the cleaned dataset for use in feature engineering and modeling.

# %%
OUTPUT_DIR = os.path.abspath(os.path.join('..', 'data', 'processed'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, 'tickets_clean.csv')
df_clean.to_csv(output_path, index=False)
print(f"✅ Cleaned data saved to: {output_path}")
print(f"   Shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

# %% [markdown]
# ---
# ## ✅ Preprocessing Complete
#
# **What we did:**
# - Downloaded required NLTK data (stopwords, punkt, wordnet)
# - Cleaned text: lowercased, removed URLs/emails/HTML/special chars,
#   removed stopwords, applied lemmatization
# - Reduced vocabulary by ~50%+ while preserving semantic content
# - Saved cleaned data to `data/processed/tickets_clean.csv`
#
# **Next Steps →** `03_feature_engineering.py` — Create TF-IDF, BoW,
# and metadata features from the cleaned text.

# %%
print("✅ Notebook 02 complete. Proceed to notebook 03.")
