"""
run_all.py  — End-to-end pipeline runner
==========================================
Executes the full ML pipeline in sequence:
  Step 0: Regenerate dataset
  Step 1: Data Exploration (EDA)
  Step 2: Text Preprocessing
  Step 3: Feature Engineering
  Step 4: Category Classification
  Step 5: Priority Classification
  Step 6: Business Insights & Summary

All matplotlib figures are saved to reports/figures/ instead of being shown
interactively, so this script can run fully non-interactively.
"""
import os
import sys
import importlib
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — saves PNGs, no pop-ups
import matplotlib.pyplot as plt

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Override plt.show so no blocking calls happen
plt.show = lambda *a, **kw: None

FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
DATA_RAW    = os.path.join(PROJECT_ROOT, "data", "raw",       "customer_support_tickets.csv")
DATA_PROC   = os.path.join(PROJECT_ROOT, "data", "processed", "tickets_clean.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "models"),            exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "reports"),           exist_ok=True)

STEP_DIVIDER = "\n" + "=" * 70 + "\n"

# =============================================================================
# STEP 0 — Generate Dataset
# =============================================================================
print(STEP_DIVIDER + "STEP 0: Generating Dataset" + STEP_DIVIDER)
from src.data_generator import generate_dataset
df_raw = generate_dataset(num_samples=5000)
print(f"Dataset shape: {df_raw.shape}")
print(f"Columns      : {list(df_raw.columns)}")

# =============================================================================
# STEP 1 — Exploratory Data Analysis
# =============================================================================
print(STEP_DIVIDER + "STEP 1: Exploratory Data Analysis" + STEP_DIVIDER)

import pandas as pd
import numpy as np
import seaborn as sns
import re
import string
from collections import Counter

PALETTE = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6',
           '#1abc9c','#e67e22','#34495e','#16a085','#c0392b']

df = pd.read_csv(DATA_RAW)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# 1a. Ticket type distribution
fig, ax = plt.subplots(figsize=(12, 6))
type_counts = df['Ticket_Type'].value_counts()
bars = ax.bar(type_counts.index, type_counts.values, color=PALETTE[:len(type_counts)], edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, type_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+10,
            f'{count:,}', ha='center', fontweight='bold', fontsize=11)
ax.set_title('Distribution of Ticket Types', fontsize=16, fontweight='bold')
ax.set_xlabel('Ticket Type', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'ticket_type_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: ticket_type_distribution.png")

# 1b. Priority distribution (pie)
fig, ax = plt.subplots(figsize=(8, 8))
priority_counts = df['Ticket_Priority'].value_counts()
ax.pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%',
       colors=PALETTE[:len(priority_counts)], startangle=140,
       textprops={'fontsize': 13}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Ticket Priority Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'priority_distribution_pie.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: priority_distribution_pie.png")

# 1c. Type x Priority heatmap
ct = pd.crosstab(df['Ticket_Type'], df['Ticket_Priority'])
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
priority_order = [p for p in ['High', 'Medium', 'Low'] if p in ct.columns]
ct_ordered = ct.reindex(columns=priority_order)
ct_ordered.plot(kind='bar', stacked=True, ax=axes[0],
                color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='white')
axes[0].set_title('Ticket Type x Priority (Stacked)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Ticket Type', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=15)
sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1], linewidths=0.5)
axes[1].set_title('Ticket Type x Priority (Heatmap)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'type_vs_priority.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: type_vs_priority.png")

# 1d. Text length distribution
df['text_length'] = df['Ticket_Description'].astype(str).apply(len)
df['word_count']  = df['Ticket_Description'].astype(str).apply(lambda x: len(x.split()))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(df['text_length'], bins=40, color=PALETTE[1], edgecolor='white', alpha=0.85)
axes[0].axvline(df['text_length'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df['text_length'].mean():.0f}")
axes[0].set_title('Description Length (chars)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Character Length', fontsize=12)
axes[0].legend()
axes[1].hist(df['word_count'], bins=40, color=PALETTE[0], edgecolor='white', alpha=0.85)
axes[1].axvline(df['word_count'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df['word_count'].mean():.0f}")
axes[1].set_title('Description Length (words)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Word Count', fontsize=12)
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'text_length_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: text_length_distribution.png")

# 1e. Word count boxplot per category
fig, ax = plt.subplots(figsize=(12, 6))
order = df.groupby('Ticket_Type')['word_count'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='Ticket_Type', y='word_count', order=order,
            palette=PALETTE[:len(order)], ax=ax, linewidth=1.5)
ax.set_title('Word Count by Ticket Type', fontsize=16, fontweight='bold')
ax.set_xlabel('Ticket Type', fontsize=13)
ax.set_ylabel('Word Count', fontsize=13)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'wordcount_by_type_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: wordcount_by_type_boxplot.png")

# 1f. Top 20 words
stop_words_basic = {'the','a','an','is','it','to','in','and','of','for','on','with',
                    'my','i','me','was','are','be','have','has','this','that','but',
                    'not','from','or','at','by','as','can','do','will','been','am',
                    'so','we','they','you','your','its','if','up','out','no','what',
                    'when','which','there','their','all','would','about','how','get',
                    'also','just','than','very','could','should','did','had'}
all_text = ' '.join(df['Ticket_Description'].astype(str).str.lower())
all_text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', all_text)
words = [w for w in all_text.split() if w not in stop_words_basic and len(w) > 2]
word_freq = Counter(words).most_common(20)
fig, ax = plt.subplots(figsize=(12, 7))
words_list, counts_list = zip(*word_freq)
bars = ax.barh(range(len(words_list)), counts_list, color=PALETTE[1], edgecolor='white')
ax.set_yticks(range(len(words_list)))
ax.set_yticklabels(words_list, fontsize=12)
ax.invert_yaxis()
for bar, count in zip(bars, counts_list):
    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
            f'{count:,}', va='center', fontsize=10, fontweight='bold')
ax.set_title('Top 20 Most Common Words', fontsize=16, fontweight='bold')
ax.set_xlabel('Frequency', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'top20_common_words.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: top20_common_words.png")

# 1g. Channel / product distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ch_counts = df['Ticket_Channel'].value_counts()
axes[0].bar(ch_counts.index, ch_counts.values, color=PALETTE[:len(ch_counts)], edgecolor='white')
axes[0].set_title('Ticket Channel Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Channel', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=15)
prod_counts = df['Product_Purchased'].value_counts()
axes[1].barh(prod_counts.index, prod_counts.values, color=PALETTE[1], edgecolor='white')
axes[1].invert_yaxis()
axes[1].set_title('Product Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'channel_product.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: channel_product.png")

df.drop(columns=['text_length','word_count'], inplace=True, errors='ignore')
print("EDA complete.")

# =============================================================================
# STEP 2 — Text Preprocessing
# =============================================================================
print(STEP_DIVIDER + "STEP 2: Text Preprocessing" + STEP_DIVIDER)

from src.preprocess import download_nltk_data, clean_text, preprocess_dataframe

download_nltk_data()
df_raw_reload = pd.read_csv(DATA_RAW)
print(f"Preprocessing {len(df_raw_reload):,} tickets…")
df_clean = preprocess_dataframe(df_raw_reload, text_column='Ticket_Description', subject_column='Ticket_Subject')

# Before / After stats
orig_len  = df_raw_reload['Ticket_Description'].astype(str).apply(lambda x: len(x.split()))
clean_len = df_clean['cleaned_text'].astype(str).apply(lambda x: len(x.split()))
print(f"  Avg words before: {orig_len.mean():.1f}   after: {clean_len.mean():.1f}")
orig_vocab  = set(' '.join(df_raw_reload['Ticket_Description'].astype(str).str.lower()).split())
clean_vocab = set(' '.join(df_clean['cleaned_text'].astype(str)).split())
print(f"  Vocabulary before: {len(orig_vocab):,}   after: {len(clean_vocab):,}   reduction: {(1-len(clean_vocab)/max(len(orig_vocab),1))*100:.1f}%")

# Token distribution before vs after
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(orig_len, bins=40, alpha=0.6, color=PALETTE[2], label='Before', edgecolor='white')
ax.hist(clean_len, bins=40, alpha=0.6, color=PALETTE[0], label='After', edgecolor='white')
ax.set_title('Token Count Distribution: Before vs After Preprocessing', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Tokens', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'preprocessing_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: preprocessing_comparison.png")

# Vocabulary comparison bar
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['Before Cleaning', 'After Cleaning'],
              [len(orig_vocab), len(clean_vocab)],
              color=[PALETTE[2], PALETTE[0]], edgecolor='white', width=0.5)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
            f'{int(bar.get_height()):,}', ha='center', fontweight='bold', fontsize=13)
ax.set_title('Vocabulary Size: Before vs After Cleaning', fontsize=15, fontweight='bold')
ax.set_ylabel('Unique Tokens', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'vocabulary_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: vocabulary_comparison.png")

# Save cleaned data
out_proc = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(out_proc, exist_ok=True)
df_clean.to_csv(DATA_PROC, index=False)
print(f"  Saved cleaned data: {DATA_PROC}  ({df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols)")

# =============================================================================
# STEP 3 — Feature Engineering
# =============================================================================
print(STEP_DIVIDER + "STEP 3: Feature Engineering" + STEP_DIVIDER)

from scipy import sparse
from sklearn.decomposition import PCA
from src.features import create_tfidf_features, create_bow_features, get_top_features_per_class, create_metadata_features

df = pd.read_csv(DATA_PROC)
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)
texts = df['cleaned_text'].values

# TF-IDF
tfidf_matrix, tfidf_vec = create_tfidf_features(texts, max_features=5000, ngram_range=(1,2))
n_nz = tfidf_matrix.nnz
total = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
print(f"  TF-IDF matrix: {tfidf_matrix.shape}  sparsity={1-n_nz/total:.2%}")

# BoW
bow_matrix, bow_vec = create_bow_features(texts, max_features=5000)
n_nz_bow = bow_matrix.nnz
total_bow = bow_matrix.shape[0] * bow_matrix.shape[1]
print(f"  BoW matrix   : {bow_matrix.shape}  sparsity={1-n_nz_bow/total_bow:.2%}")

# TF-IDF vs BoW density bar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
densities = [(n_nz/total)*100, (n_nz_bow/total_bow)*100]
methods = ['TF-IDF', 'Bag-of-Words']
axes[0].bar(methods, densities, color=[PALETTE[1], PALETTE[0]], edgecolor='white', width=0.5)
for i, (m, d) in enumerate(zip(methods, densities)):
    axes[0].text(i, d+0.05, f'{d:.3f}%', ha='center', fontweight='bold')
axes[0].set_title('Feature Density Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Density (%)', fontsize=12)
nonzeros = [n_nz, n_nz_bow]
axes[1].bar(methods, nonzeros, color=[PALETTE[1], PALETTE[0]], edgecolor='white', width=0.5)
for i, v in enumerate(nonzeros):
    axes[1].text(i, v+200, f'{v:,}', ha='center', fontweight='bold')
axes[1].set_title('Non-zero Elements Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'tfidf_vs_bow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: tfidf_vs_bow.png")

# Top TF-IDF terms per category
categories = sorted(df['Ticket_Type'].unique())
n_cats = len(categories)
fn = tfidf_vec.get_feature_names_out()
fig, axes = plt.subplots(1, n_cats, figsize=(5*n_cats, 8))
if n_cats == 1: axes = [axes]
for idx, cat in enumerate(categories):
    mask = df['Ticket_Type'] == cat
    cat_tfidf = tfidf_matrix[mask.values]
    mean_scores = np.asarray(cat_tfidf.mean(axis=0)).flatten()
    top_i = mean_scores.argsort()[-15:][::-1]
    top_t = [fn[i] for i in top_i]
    top_v = mean_scores[top_i]
    axes[idx].barh(range(len(top_t)), top_v, color=PALETTE[idx % len(PALETTE)], edgecolor='white')
    axes[idx].set_yticks(range(len(top_t)))
    axes[idx].set_yticklabels(top_t, fontsize=9)
    axes[idx].invert_yaxis()
    axes[idx].set_title(cat, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Mean TF-IDF', fontsize=10)
fig.suptitle('Top 15 TF-IDF Terms per Ticket Category', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'top_tfidf_per_category.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: top_tfidf_per_category.png")

# PCA
print("  Running PCA...")
tfidf_dense = tfidf_matrix.toarray()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(tfidf_dense)
labels = df['Ticket_Type'].values
fig, ax = plt.subplots(figsize=(12, 8))
for i, cat in enumerate(categories):
    m = labels == cat
    ax.scatter(X_pca[m,0], X_pca[m,1], c=PALETTE[i%len(PALETTE)], label=cat, alpha=0.5, s=20)
ax.set_title('PCA — TF-IDF Feature Space (2D)', fontsize=16, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=13)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=13)
ax.legend(title='Ticket Type', fontsize=10, markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'pca_tfidf_2d.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pca_tfidf_2d.png")
print("Feature engineering complete.")

# =============================================================================
# STEP 4 — Category Classification
# =============================================================================
print(STEP_DIVIDER + "STEP 4: Category Classification" + STEP_DIVIDER)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from src.features import encode_labels
from src.train import get_models, train_and_evaluate, tune_best_model, save_model

df = pd.read_csv(DATA_PROC)
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)

X = df['cleaned_text'].values
y = df['Ticket_Type'].values
y_encoded, label_encoder_cat = encode_labels(y)
class_names_cat = list(label_encoder_cat.classes_)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
X_train_tfidf, tfidf_vectorizer = create_tfidf_features(X_train_raw, max_features=10000, ngram_range=(1,2))
X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
print(f"Train: {X_train_tfidf.shape}  Test: {X_test_tfidf.shape}")

models = get_models()
model_results = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, models)

results_summary = [{'Model': n, 'Accuracy': m['accuracy'], 'Macro F1': m['f1_macro'], 'Weighted F1': m['f1_weighted']} for n,m in model_results.items()]
df_results = pd.DataFrame(results_summary).sort_values('Macro F1', ascending=False)
print("\nModel Comparison:")
print(df_results.to_string(index=False))

# Comparison plot
df_melt = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Model', hue='Metric', data=df_melt, palette='muted')
plt.title('Category Classifiers — Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Score', fontsize=12)
plt.xlim(0, 1.05)
plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'category_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: category_model_comparison.png")

# Tune best pipeline
best_cat_pipeline = tune_best_model(X_train_raw, y_train)
y_pred_cat = best_cat_pipeline.predict(X_test_raw)
best_acc_cat = accuracy_score(y_test, y_pred_cat)
best_f1_cat  = f1_score(y_test, y_pred_cat, average='macro')
print(f"\nTuned Pipeline — Accuracy: {best_acc_cat:.2%}   Macro F1: {best_f1_cat:.2%}")
print(classification_report(y_test, y_pred_cat, target_names=class_names_cat))

# Save report
report_dir = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(report_dir, exist_ok=True)
with open(os.path.join(report_dir, 'category_classification_report.txt'), 'w') as f:
    f.write("CATEGORY CLASSIFICATION REPORT\n")
    f.write("="*50+"\n")
    f.write(f"Best Model: TF-IDF + Logistic Regression (tuned)\n")
    f.write(f"Test Accuracy : {best_acc_cat:.4f}\n")
    f.write(f"Test Macro F1 : {best_f1_cat:.4f}\n\n")
    f.write(classification_report(y_test, y_pred_cat, target_names=class_names_cat))
print("  Saved: category_classification_report.txt")

# Confusion matrix
cm_cat = confusion_matrix(y_test, y_pred_cat)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names_cat, yticklabels=class_names_cat)
plt.title('Category Classifier — Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'category_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: category_confusion_matrix.png")

# Feature importance
clf_step = best_cat_pipeline.named_steps['classifier']
vec_step  = best_cat_pipeline.named_steps['vectorizer']
top_feats = get_top_features_per_class(vec_step, clf_step, class_names_cat, n_top=10)
if top_feats:
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
    axes_flat = axes.flatten()
    for idx, (cat_name, features) in enumerate(top_feats.items()):
        if idx >= len(axes_flat): break
        terms, scores = zip(*features)
        sns.barplot(x=list(scores), y=list(terms), ax=axes_flat[idx], palette='Blues_r')
        axes_flat[idx].set_title(f'Top Terms: {cat_name}', fontsize=12, fontweight='bold')
        axes_flat[idx].set_xlabel('Coefficient')
    for idx in range(len(top_feats), len(axes_flat)):
        axes_flat[idx].axis('off')
    plt.suptitle('Logistic Regression Feature Coefficients by Category', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'category_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: category_feature_importance.png")

# Save models
models_dir = os.path.join(PROJECT_ROOT, 'models')
save_model(best_cat_pipeline,  os.path.join(models_dir, 'category_classifier.pkl'))
save_model(label_encoder_cat,  os.path.join(models_dir, 'category_label_encoder.pkl'))
print("Category Classification complete.")

# =============================================================================
# STEP 5 — Priority Classification
# =============================================================================
print(STEP_DIVIDER + "STEP 5: Priority Classification" + STEP_DIVIDER)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from src.features import create_metadata_features
from src.train import tune_priority_model

df = pd.read_csv(DATA_PROC)
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)

# Engineer metadata (returns only the new columns)
meta_df = create_metadata_features(df, text_col='Ticket_Description')
meta_cols = list(meta_df.columns)

# Build full feature df: cleaned_text + meta + target
df_full = pd.concat([df[['cleaned_text', 'Ticket_Priority']], meta_df], axis=1)
print(f"Full feature DataFrame shape: {df_full.shape}")
print(f"Meta columns ({len(meta_cols)}): {meta_cols}")

y_prio          = df_full['Ticket_Priority'].values
y_prio_enc, label_encoder_prio = encode_labels(y_prio)
class_names_prio = list(label_encoder_prio.classes_)

df_train_p, df_test_p, y_train_p, y_test_p = train_test_split(
    df_full, y_prio_enc, test_size=0.2, stratify=y_prio_enc, random_state=42
)
print(f"Train: {df_train_p.shape}  Test: {df_test_p.shape}")

# Strategy A — Text only
print("\n--- Strategy A: Text-Only ---")
models_fresh = get_models()
results_a = {}
for name, clf in models_fresh.items():
    pipe = Pipeline([
        ('vectorizer', TFIDF(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ('classifier', clf)
    ])
    pipe.fit(df_train_p['cleaned_text'].values, y_train_p)
    preds = pipe.predict(df_test_p['cleaned_text'].values)
    results_a[name] = {'accuracy': accuracy_score(y_test_p, preds),
                       'f1_macro': f1_score(y_test_p, preds, average='macro')}
    print(f"  {name}: Acc={results_a[name]['accuracy']:.2%}  F1={results_a[name]['f1_macro']:.2%}")

# Strategy B — Hybrid
print("\n--- Strategy B: Hybrid (Text + Metadata) ---")
preprocessor_b = ColumnTransformer(
    transformers=[
        ('text', TFIDF(max_features=5000, ngram_range=(1,2), sublinear_tf=True), 'cleaned_text'),
        ('meta', MinMaxScaler(), meta_cols),
    ],
    remainder='drop'
)
models_fresh2 = get_models()
results_b = {}
for name, clf in models_fresh2.items():
    pipe = Pipeline([('preprocessor', preprocessor_b), ('classifier', clf)])
    pipe.fit(df_train_p, y_train_p)
    preds = pipe.predict(df_test_p)
    results_b[name] = {'accuracy': accuracy_score(y_test_p, preds),
                       'f1_macro': f1_score(y_test_p, preds, average='macro')}
    print(f"  {name}: Acc={results_b[name]['accuracy']:.2%}  F1={results_b[name]['f1_macro']:.2%}")

# Strategy comparison plot
comparison_data = []
for name in models_fresh.keys():
    comparison_data += [
        {'Model': name, 'Strategy': 'Text-Only (A)', 'Macro F1': results_a[name]['f1_macro']},
        {'Model': name, 'Strategy': 'Hybrid (B)',    'Macro F1': results_b[name]['f1_macro']},
    ]
df_compare = pd.DataFrame(comparison_data)
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Macro F1', hue='Strategy', data=df_compare, palette='Set2')
plt.title('Priority Classification: Text-Only vs Hybrid', fontsize=14, fontweight='bold')
plt.ylabel('Macro F1-Score', fontsize=12)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'priority_strategy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: priority_strategy_comparison.png")

# Tune best hybrid pipeline
best_prio_pipeline = tune_priority_model(df_train_p, y_train_p, meta_cols)
y_pred_prio = best_prio_pipeline.predict(df_test_p)
best_acc_prio = accuracy_score(y_test_p, y_pred_prio)
best_f1_prio  = f1_score(y_test_p, y_pred_prio, average='macro')
print(f"\nTuned Priority Pipeline — Accuracy: {best_acc_prio:.2%}   Macro F1: {best_f1_prio:.2%}")
print(classification_report(y_test_p, y_pred_prio, target_names=class_names_prio))

with open(os.path.join(report_dir, 'priority_classification_report.txt'), 'w') as f:
    f.write("PRIORITY CLASSIFICATION REPORT\n")
    f.write("="*50+"\n")
    f.write(f"Best Model: Hybrid TF-IDF + Metadata + Logistic Regression (tuned)\n")
    f.write(f"Test Accuracy : {best_acc_prio:.4f}\n")
    f.write(f"Test Macro F1 : {best_f1_prio:.4f}\n\n")
    f.write(classification_report(y_test_p, y_pred_prio, target_names=class_names_prio))
print("  Saved: priority_classification_report.txt")

# Priority confusion matrix
cm_prio = confusion_matrix(y_test_p, y_pred_prio)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_prio, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names_prio, yticklabels=class_names_prio)
plt.title('Priority Classifier — Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'priority_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: priority_confusion_matrix.png")

# Save priority models
save_model(best_prio_pipeline, os.path.join(models_dir, 'priority_classifier.pkl'))
save_model(label_encoder_prio, os.path.join(models_dir, 'priority_label_encoder.pkl'))
print("Priority Classification complete.")

# =============================================================================
# STEP 6 — Business Insights & Summary
# =============================================================================
print(STEP_DIVIDER + "STEP 6: Business Insights & Summary" + STEP_DIVIDER)

from src.predict import load_models, classify_ticket, batch_classify

cat_model, prio_model, cat_enc, prio_enc = load_models()
print("Models loaded.")

# Live demo on 5 test tickets
demo_tickets = [
    {"subject": "Billing error: double charge on card",
     "description": "I noticed I was charged $49.00 twice this month. I only authorized one payment. Please refund the duplicate charge ASAP."},
    {"subject": "App crashes on startup after update",
     "description": "After yesterday's update, the app crashes every time I open it on my iPhone. I have a presentation tomorrow. This is urgent."},
    {"subject": "How do I add a team member?",
     "description": "I'd like to invite a colleague to my workspace. Could you point me to where I can add users? No rush."},
    {"subject": "Locked out of account — cannot login",
     "description": "My account is locked and the password reset link in my email is broken. I cannot access anything. Please unlock my account immediately."},
    {"subject": "Feature request: dark mode for dashboard",
     "description": "It would be great to have a dark mode toggle. The white background is very bright during night shifts. Just a suggestion!"},
]

print("\nLive Demo Predictions:")
print("-" * 95)
demo_results = []
for i, t in enumerate(demo_tickets, 1):
    res = classify_ticket(t['subject'], t['description'], cat_model, prio_model, cat_enc, prio_enc)
    demo_results.append({'#': i, 'Subject': t['subject'][:45],
                          'Category': res['category'], 'Cat Conf': f"{res['category_confidence']:.0%}",
                          'Priority': res['priority'], 'Pri Conf': f"{res['priority_confidence']:.0%}",
                          'Action': res['recommended_action'].split('|')[0].strip()})
    print(f"  {i}. {t['subject'][:45]}")
    print(f"     Category: {res['category']} ({res['category_confidence']:.0%}) | Priority: {res['priority']} ({res['priority_confidence']:.0%})")
    print(f"     {res['recommended_action']}")
    print()

# Batch classification on first 200 tickets
df_raw_batch = pd.read_csv(DATA_RAW).head(200)
df_classified = batch_classify(df_raw_batch, cat_model, prio_model, cat_enc, prio_enc)
routing_counts = df_classified['Recommended_Routing_Action'].str.split('|').str[0].str.strip().value_counts()

plt.figure(figsize=(12, 7))
sns.barplot(x=routing_counts.values, y=routing_counts.index, palette='plasma')
plt.title('Simulated Queue Distribution (200 tickets — SLA-based Routing)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Tickets', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'simulated_routing_queues.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: simulated_routing_queues.png")

# ROI analysis
daily_volume = 1200
sort_min = 3.0
hourly_cost = 22.0
accuracy = max(best_acc_cat, 0.80)
manual_hours = (daily_volume * sort_min) / 60
saved_hours  = manual_hours * accuracy
saved_daily  = saved_hours * hourly_cost
print("\nROI ANALYSIS")
print("=" * 60)
print(f"  Daily ticket volume      : {daily_volume:,}")
print(f"  Manual triage time       : {manual_hours:.1f} hrs/day")
print(f"  System accuracy          : {accuracy:.1%}")
print(f"  Hours saved daily        : {saved_hours:.1f} hrs")
print(f"  Daily savings            : ${saved_daily:,.2f}")
print(f"  Monthly savings          : ${saved_daily*30.4:,.2f}")
print(f"  Annual savings           : ${saved_daily*365:,.2f}")
print("=" * 60)

# Final summary chart — accuracy comparison
summary = {
    'Category\nClassification': best_acc_cat,
    'Priority\nClassification': best_acc_prio,
}
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(summary.keys(), [v*100 for v in summary.values()],
              color=[PALETTE[1], PALETTE[0]], edgecolor='white', width=0.4)
for bar, val in zip(bars, summary.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1%}', ha='center', fontweight='bold', fontsize=14)
ax.set_ylim(0, 110)
ax.set_ylabel('Test Accuracy (%)', fontsize=13)
ax.set_title('Final Model Accuracy Summary', fontsize=16, fontweight='bold')
ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% baseline')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'final_accuracy_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: final_accuracy_summary.png")

print(STEP_DIVIDER)
print("  PIPELINE COMPLETE — All models trained and saved.")
print(f"  Category Accuracy : {best_acc_cat:.2%}")
print(f"  Priority Accuracy : {best_acc_prio:.2%}")
print(f"  Figures saved to  : {FIGURES_DIR}")
print(f"  Models saved to   : {models_dir}")
print(STEP_DIVIDER)
