# %% [markdown]
# # ⚙️ 03 — Feature Engineering
# **Customer Support Ticket Classification Project**
#
# This notebook creates and analyzes the feature representations used for
# ticket classification. We build TF-IDF and Bag-of-Words features, engineer
# metadata features, and visualize feature quality using PCA/t-SNE.
#
# **Sections:**
# 1. Setup & Data Loading
# 2. TF-IDF Features
# 3. Bag-of-Words Features
# 4. TF-IDF vs BoW Comparison
# 5. Top TF-IDF Terms per Category
# 6. Metadata Features
# 7. Correlation Heatmap
# 8. Dimensionality Reduction (PCA / t-SNE)
# 9. Feature Summary

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
from scipy import sparse

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
PALETTE = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6',
           '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
sns.set_palette(PALETTE)

FIGDIR = os.path.abspath(os.path.join('..', 'reports', 'figures'))
os.makedirs(FIGDIR, exist_ok=True)

from src.features import (create_tfidf_features, create_bow_features,
                           encode_labels, get_top_features_per_class,
                           create_metadata_features)

# %% [markdown]
# ---
# ## 1 · Load Cleaned Data

# %%
DATA_PATH = os.path.join('..', 'data', 'processed', 'tickets_clean.csv')
df = pd.read_csv(DATA_PATH)
print(f"Loaded cleaned dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

# %%
# Identify the cleaned text column
text_candidates = ['cleaned_text', 'clean_text', 'processed_text', 'Ticket_Description']
TEXT_COL = None
for col in text_candidates:
    if col in df.columns:
        TEXT_COL = col
        break

if TEXT_COL is None:
    raise ValueError(f"No text column found. Available: {list(df.columns)}")

print(f"Using text column: '{TEXT_COL}'")
print(f"Sample:\n  \"{df[TEXT_COL].iloc[0][:120]}...\"")

# Fill any NaN values in text column
df[TEXT_COL] = df[TEXT_COL].fillna('')

# %% [markdown]
# ---
# ## 2 · TF-IDF Features
# Create Term Frequency–Inverse Document Frequency features, the primary
# text representation for our classifiers.

# %%
texts = df[TEXT_COL].values
tfidf_matrix, tfidf_vectorizer = create_tfidf_features(
    texts, max_features=5000, ngram_range=(1, 2)
)

print("TF-IDF Feature Matrix")
print(f"  Shape:    {tfidf_matrix.shape}")
print(f"  Type:     {type(tfidf_matrix).__name__}")
print(f"  Dtype:    {tfidf_matrix.dtype}")
n_nonzero = tfidf_matrix.nnz if sparse.issparse(tfidf_matrix) else np.count_nonzero(tfidf_matrix)
total_elements = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
sparsity = 1.0 - (n_nonzero / total_elements)
print(f"  Non-zero: {n_nonzero:,} / {total_elements:,}")
print(f"  Sparsity: {sparsity * 100:.2f}%")
print(f"  Memory:   {tfidf_matrix.data.nbytes / 1024 / 1024:.2f} MB (data only)")

# %%
# Show some feature names
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"\nTotal features: {len(feature_names):,}")
print(f"First 20 features: {list(feature_names[:20])}")
print(f"Last 20 features:  {list(feature_names[-20:])}")

# %% [markdown]
# ---
# ## 3 · Bag-of-Words Features
# Create a simple count-based representation for comparison.

# %%
bow_matrix, bow_vectorizer = create_bow_features(texts, max_features=5000)

print("Bag-of-Words Feature Matrix")
print(f"  Shape:    {bow_matrix.shape}")
n_nonzero_bow = bow_matrix.nnz if sparse.issparse(bow_matrix) else np.count_nonzero(bow_matrix)
total_bow = bow_matrix.shape[0] * bow_matrix.shape[1]
sparsity_bow = 1.0 - (n_nonzero_bow / total_bow)
print(f"  Non-zero: {n_nonzero_bow:,} / {total_bow:,}")
print(f"  Sparsity: {sparsity_bow * 100:.2f}%")

# %% [markdown]
# ---
# ## 4 · TF-IDF vs BoW Comparison
# Compare the density and properties of both feature representations.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sparsity comparison
methods = ['TF-IDF', 'Bag-of-Words']
sparsities = [sparsity * 100, sparsity_bow * 100]
densities = [100 - s for s in sparsities]

bars = axes[0].bar(methods, densities, color=[PALETTE[1], PALETTE[0]],
                   edgecolor='white', width=0.5)
for bar, val in zip(bars, densities):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.2f}%', ha='center', fontweight='bold', fontsize=12)
axes[0].set_title('Feature Density Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Density (%)', fontsize=12)
axes[0].set_ylim(0, max(densities) * 1.3)

# Non-zero elements comparison
nonzeros = [n_nonzero, n_nonzero_bow]
bars2 = axes[1].bar(methods, nonzeros, color=[PALETTE[1], PALETTE[0]],
                    edgecolor='white', width=0.5)
for bar, val in zip(bars2, nonzeros):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f'{val:,}', ha='center', fontweight='bold', fontsize=12)
axes[1].set_title('Non-zero Elements Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'tfidf_vs_bow.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📁 Saved: tfidf_vs_bow.png")

# %% [markdown]
# ---
# ## 5 · Top TF-IDF Terms per Category
# For each ticket category, identify the terms with the highest average
# TF-IDF weight. This reveals the discriminative vocabulary for each class.

# %%
categories = sorted(df['Ticket_Type'].unique())
n_cats = len(categories)

fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 8), sharey=False)
if n_cats == 1:
    axes = [axes]

for idx, cat in enumerate(categories):
    mask = df['Ticket_Type'] == cat
    cat_tfidf = tfidf_matrix[mask.values]

    # Mean TF-IDF score per term for this category
    mean_scores = np.asarray(cat_tfidf.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[-20:][::-1]
    top_terms = [feature_names[i] for i in top_indices]
    top_values = mean_scores[top_indices]

    axes[idx].barh(range(len(top_terms)), top_values, color=PALETTE[idx % len(PALETTE)],
                   edgecolor='white')
    axes[idx].set_yticks(range(len(top_terms)))
    axes[idx].set_yticklabels(top_terms, fontsize=10)
    axes[idx].invert_yaxis()
    axes[idx].set_title(f'{cat}', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Mean TF-IDF Score', fontsize=10)

fig.suptitle('Top 20 TF-IDF Terms per Ticket Category', fontsize=16,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'top_tfidf_per_category.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: top_tfidf_per_category.png")

# %% [markdown]
# ---
# ## 6 · Metadata Features
# Engineer additional numeric features from the text and other columns:
# text length, word count, exclamation marks, question marks, etc.

# %%
metadata_df = create_metadata_features(df)

# Show the metadata columns that were created
meta_cols = [c for c in metadata_df.columns if c not in df.columns]
print(f"Metadata features created ({len(meta_cols)}):")
for col in meta_cols:
    print(f"  • {col}: {metadata_df[col].dtype} "
          f"(mean={metadata_df[col].mean():.2f}, std={metadata_df[col].std():.2f})")

# %%
metadata_df[meta_cols].describe().round(2)

# %% [markdown]
# ---
# ## 7 · Correlation Heatmap
# Examine the pairwise correlations among metadata features to identify
# redundant or highly correlated variables.

# %%
if len(meta_cols) >= 2:
    corr = metadata_df[meta_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Metadata Feature Correlations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'metadata_correlation.png'), dpi=150,
                bbox_inches='tight')
    plt.show()
    print("📁 Saved: metadata_correlation.png")
else:
    print("⏭️  Not enough metadata features for correlation analysis.")

# %% [markdown]
# ---
# ## 8 · Dimensionality Reduction (PCA / t-SNE)
# Reduce the high-dimensional TF-IDF feature space to 2D for visualization.
# This helps us see whether ticket categories form distinct clusters.

# %%
from sklearn.decomposition import PCA

print("Running PCA (TF-IDF → 2D)...")
pca = PCA(n_components=2, random_state=42)
tfidf_dense = tfidf_matrix.toarray() if sparse.issparse(tfidf_matrix) else tfidf_matrix
X_pca = pca.fit_transform(tfidf_dense)

print(f"  Explained variance: Component 1 = {pca.explained_variance_ratio_[0]:.3f}, "
      f"Component 2 = {pca.explained_variance_ratio_[1]:.3f}")
print(f"  Total explained:    {sum(pca.explained_variance_ratio_):.3f}")

# %%
fig, ax = plt.subplots(figsize=(12, 8))
labels = df['Ticket_Type'].values

for i, cat in enumerate(categories):
    mask = labels == cat
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=PALETTE[i % len(PALETTE)],
               label=cat, alpha=0.5, s=20, edgecolors='none')

ax.set_title('PCA — TF-IDF Feature Space (2D Projection)', fontsize=16,
             fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=13)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=13)
ax.legend(title='Ticket Type', fontsize=10, title_fontsize=11, markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'pca_tfidf_2d.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📁 Saved: pca_tfidf_2d.png")

# %%
try:
    from sklearn.manifold import TSNE

    print("Running t-SNE (TF-IDF → 2D)... this may take a minute...")

    # Use PCA pre-reduction for speed if dataset is large
    n_pca_components = min(50, tfidf_matrix.shape[1])
    pca_pre = PCA(n_components=n_pca_components, random_state=42)
    X_pca_pre = pca_pre.fit_transform(tfidf_dense)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000,
                learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(X_pca_pre)

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, cat in enumerate(categories):
        mask = labels == cat
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=PALETTE[i % len(PALETTE)],
                   label=cat, alpha=0.5, s=20, edgecolors='none')

    ax.set_title('t-SNE — TF-IDF Feature Space (2D Projection)', fontsize=16,
                 fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.legend(title='Ticket Type', fontsize=10, title_fontsize=11, markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'tsne_tfidf_2d.png'), dpi=150,
                bbox_inches='tight')
    plt.show()
    print("📁 Saved: tsne_tfidf_2d.png")

except ImportError:
    print("⏭️  t-SNE not available.")

# %% [markdown]
# ---
# ## 9 · Feature Summary
#
# | Feature Set | Dimensions | Sparsity | Notes |
# |------------|-----------|---------|-------|
# | **TF-IDF** | (n_samples, 5000) | ~97%+ | Primary features for classification |
# | **Bag-of-Words** | (n_samples, 5000) | ~97%+ | Baseline comparison |
# | **Metadata** | (n_samples, k) | 0% (dense) | Supplementary signals |
#
# **Key Observations:**
# - TF-IDF captures category-specific vocabulary well (visible in top terms)
# - PCA/t-SNE show partial but imperfect cluster separation — models needed
# - Metadata features (text length, punctuation counts) add complementary signal
# - Feature matrices are highly sparse — suited for linear models (SVM, LogReg)
#
# **Next Steps →** `04_category_classification.py` — Train and evaluate
# classifiers using these features.

# %%
print("✅ Feature engineering complete. Proceed to notebook 04.")
