# %% [markdown]
# # 🚨 05 — Priority Classification Model
# **Customer Support Ticket Classification Project**
#
# This notebook builds and evaluates classifiers to predict the priority level
# (`Ticket_Priority`: High / Medium / Low) of support tickets. We compare two approaches:
# - **Strategy A (Text-only):** TF-IDF features of the ticket text alone.
# - **Strategy B (Hybrid):** Combines TF-IDF text features with engineered metadata indicators 
#   (lengths, exclamation marks, urgent keywords, etc.).
#
# **Sections:**
# 1. Setup & Data Loading
# 2. Metadata Engineering & Dataset Split
# 3. Strategy A: Text-Only Model Performance
# 4. Strategy B: Hybrid Model Performance
# 5. Model Comparison
# 6. Detailed Evaluation of the Best Priority Model
# 7. Model Serialization

# %%
# ── Setup ────────────────────────────────────────────────────────────────────
import sys
import os
import warnings

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'H:/future interns/2'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Color palette definition
PALETTE = ['#e74c3c', '#f39c12', '#3498db']
sns.set_palette(PALETTE)

FIGDIR = os.path.join(project_root, 'reports', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
MODELS_DIR = os.path.join(project_root, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

from src.features import create_metadata_features, encode_labels
from src.train import get_models, save_model

# %% [markdown]
# ---
# ## 1 · Load Preprocessed Data

# %%
PROCESSED_DATA_PATH = os.path.join(project_root, 'data', 'processed', 'tickets_clean.csv')
df = pd.read_csv(PROCESSED_DATA_PATH)
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)
print(f"Loaded {df.shape[0]:,} rows.")

# %% [markdown]
# ---
# ## 2 · Feature Engineering (Metadata) & Data Splitting

# %%
# Extract engineered metadata features from raw df
meta_df = create_metadata_features(df)
print("Engineered metadata columns:")
print(meta_df.head())

# Combine metadata back with cleaned text and targets
df_full = pd.concat([df[['cleaned_text', 'Ticket_Priority']], meta_df], axis=1)

# Encode targets ('High', 'Medium', 'Low')
y = df_full['Ticket_Priority'].values
y_encoded, label_encoder = encode_labels(y)
class_names = list(label_encoder.classes_)

# Split dataset into stratified train/test partitions
df_train, df_test, y_train, y_test = train_test_split(
    df_full, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print(f"\nTrain set shape: {df_train.shape}")
print(f"Test set shape : {df_test.shape}")
print(f"Priority labels order: {class_names}")

# %% [markdown]
# ---
# ## 3 · Strategy A: Text-Only Model (Baseline)
# Build a pipeline using TF-IDF vectors of the ticket text alone.

# %%
print("Evaluating Strategy A: Text-Only Pipeline ...")
models = get_models()

results_a = {}
for name, clf in models.items():
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ('classifier', clf)
    ])
    pipeline.fit(df_train['cleaned_text'].values, y_train)
    preds = pipeline.predict(df_test['cleaned_text'].values)
    
    results_a[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'f1_macro': f1_score(y_test, preds, average='macro'),
        'model': pipeline
    }
    print(f"  {name} - Accuracy: {results_a[name]['accuracy']:.2%}, Macro F1: {results_a[name]['f1_macro']:.2%}")

# %% [markdown]
# ---
# ## 4 · Strategy B: Hybrid Model (Text + Metadata)
# Combine text representations and metadata features. We use `ColumnTransformer` to 
# cleanly combine preprocessing.

# %%
print("Evaluating Strategy B: Hybrid Features Pipeline ...")

# Metadata column names
meta_cols = list(meta_df.columns)

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True), 'cleaned_text'),
        ('meta', MinMaxScaler(), meta_cols)
    ]
)

results_b = {}
for name, clf in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipeline.fit(df_train, y_train)
    preds = pipeline.predict(df_test)
    
    results_b[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'f1_macro': f1_score(y_test, preds, average='macro'),
        'model': pipeline
    }
    print(f"  {name} - Accuracy: {results_b[name]['accuracy']:.2%}, Macro F1: {results_b[name]['f1_macro']:.2%}")

# %% [markdown]
# ---
# ## 5 · Model Comparison

# %%
comparison_data = []
for name in models.keys():
    comparison_data.append({
        'Model': name,
        'Strategy': 'Text-Only (A)',
        'Accuracy': results_a[name]['accuracy'],
        'Macro F1': results_a[name]['f1_macro']
    })
    comparison_data.append({
        'Model': name,
        'Strategy': 'Hybrid (B)',
        'Accuracy': results_b[name]['accuracy'],
        'Macro F1': results_b[name]['f1_macro']
    })
df_compare = pd.DataFrame(comparison_data)

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Macro F1', hue='Strategy', data=df_compare, palette='Set2')
plt.title('Priority Classification: Strategy A (Text Only) vs Strategy B (Hybrid)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Macro F1-Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0, 1.05)
plt.legend(loc='upper right')
plt.tight_layout()

compare_path = os.path.join(FIGDIR, 'priority_strategy_comparison.png')
plt.savefig(compare_path, dpi=300)
plt.show()
print(f"Strategy comparison chart saved to: {compare_path}")

# %% [markdown]
# ---
# ## 6 · Detailed Evaluation of the Best Priority Model
# Let's tune the best performing hybrid pipeline (usually Logistic Regression or Linear SVC).

# %%
# Define hyperparameter grid for tuning Logistic Regression Hybrid Model
tune_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', models['Logistic Regression'])
])

param_grid = {
    'classifier__C': [0.1, 1.0, 5.0, 10.0]
}

print("Tuning Hybrid Logistic Regression model...")
grid_search = GridSearchCV(tune_pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(df_train, y_train)

best_prio_pipeline = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")

# Evaluate on test set
y_pred = best_prio_pipeline.predict(df_test)
best_acc = accuracy_score(y_test, y_pred)
best_f1_macro = f1_score(y_test, y_pred, average='macro')

print("=" * 60)
print(f"Tuned Hybrid Priority Pipeline Test Performance:")
print(f"  Accuracy  : {best_acc:.2%}")
print(f"  Macro F1  : {best_f1_macro:.2%}")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# %%
# ── Confusion Matrix Heatmap ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Priority Classifier Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Actual Priority', fontsize=12)
plt.xlabel('Predicted Priority', fontsize=12)
plt.tight_layout()

cm_path = os.path.join(FIGDIR, 'priority_confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Confusion matrix saved to: {cm_path}")

# %%
# ── Metadata Feature Importance analysis ──────────────────────────────────────
# Inspect how coefficient weights are distributed
classifier = best_prio_pipeline.named_steps['classifier']
if hasattr(classifier, 'coef_'):
    # Extract meta coefficients (which are scaled at the end of the transformer features)
    # Total features = text sparse features (e.g. 5000) + metadata features (6)
    n_features = classifier.coef_.shape[1]
    meta_coefs = classifier.coef_[:, -len(meta_cols):]
    
    df_meta_coef = pd.DataFrame(
        meta_coefs.T, 
        index=meta_cols, 
        columns=[f"Coef_{cls}" for cls in class_names]
    )
    print("\nMetadata Coefficients across Priority Classes:")
    print(df_meta_coef)
    
    # Plot meta feature importance across classes
    df_meta_coef_plot = df_meta_coef.reset_index().rename(columns={'index': 'Feature'})
    df_meta_coef_melt = df_meta_coef_plot.melt(id_vars='Feature', var_name='Class', value_name='Weight')
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Weight', y='Feature', hue='Class', data=df_meta_coef_melt, palette='Set1')
    plt.title('Metadata Features Influence on Support Ticket Priority', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Coefficient weight')
    plt.ylabel('Engineered Feature')
    plt.tight_layout()
    
    feat_prio_path = os.path.join(FIGDIR, 'priority_meta_importance.png')
    plt.savefig(feat_prio_path, dpi=300)
    plt.show()
    print(f"Metadata feature influence plot saved to: {feat_prio_path}")

# %% [markdown]
# ---
# ## 8 · Model Serialization
# Save the final priority classification pipeline and label encoder to models/ directory.

# %%
prio_model_path = os.path.join(MODELS_DIR, 'priority_classifier.pkl')
label_encoder_path = os.path.join(MODELS_DIR, 'priority_label_encoder.pkl')

save_model(best_prio_pipeline, prio_model_path)
save_model(label_encoder, label_encoder_path)

print("\n🎉 Priority Classification modeling phase complete!")
