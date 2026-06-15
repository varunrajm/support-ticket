# %% [markdown]
# # 🎯 04 — Category Classification Model
# **Customer Support Ticket Classification Project**
#
# This notebook builds and evaluates multiple machine learning classifiers to predict the
# category of support tickets (`Ticket_Type`). We compare Naive Bayes, Logistic Regression,
# Linear Support Vector Classifier, and Random Forest, tune the best performing model,
# and serialize the final pipeline.
#
# **Sections:**
# 1. Setup & Data Loading
# 2. Train-Test Split & Vectorization
# 3. Model Training & Evaluation
# 4. Model Comparison
# 5. Hyperparameter Tuning
# 6. Detailed Evaluation of the Best Model
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
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Color palette definition
PALETTE = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
sns.set_palette(PALETTE)

FIGDIR = os.path.join(project_root, 'reports', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
MODELS_DIR = os.path.join(project_root, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

from src.features import create_tfidf_features, encode_labels, get_top_features_per_class
from src.train import get_models, train_and_evaluate, tune_best_model, save_model

# %% [markdown]
# ---
# ## 1 · Load Preprocessed Data

# %%
PROCESSED_DATA_PATH = os.path.join(project_root, 'data', 'processed', 'tickets_clean.csv')
RAW_DATA_PATH = os.path.join(project_root, 'data', 'raw', 'customer_support_tickets.csv')

if os.path.exists(PROCESSED_DATA_PATH):
    print(f"Loading cleaned dataset from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
else:
    print(f"⚠️ Cleaned dataset not found. Loading raw dataset from {RAW_DATA_PATH} and preprocessing...")
    from src.preprocess import preprocess_dataframe, download_nltk_data
    download_nltk_data()
    raw_df = pd.read_csv(RAW_DATA_PATH)
    df = preprocess_dataframe(raw_df)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

# Make sure all cleaned text entries are strings
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)
print(f"Loaded {df.shape[0]:,} rows.")

# %% [markdown]
# ---
# ## 2 · Train-Test Split & Label Encoding

# %%
# Target variable: Ticket_Type (Billing Issue, Technical Issue, Account Access, etc.)
X = df['cleaned_text'].values
y = df['Ticket_Type'].values

# Encode categorical labels to integers
y_encoded, label_encoder = encode_labels(y)
class_names = list(label_encoder.classes_)

# Perform Stratified Train-Test Split (80% Train, 20% Test)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print(f"Training set size : {X_train_raw.shape[0]:,}")
print(f"Testing set size  : {X_test_raw.shape[0]:,}")
print(f"Target classes    : {class_names}")

# %% [markdown]
# ---
# ## 3 · Text Vectorization (TF-IDF)

# %%
# Fit vectorizer on training data and transform both sets
X_train, tfidf_vectorizer = create_tfidf_features(X_train_raw, max_features=10000, ngram_range=(1, 2))
X_test = tfidf_vectorizer.transform(X_test_raw)

print(f"TF-IDF matrix vocabulary size: {len(tfidf_vectorizer.vocabulary_):,}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

# %% [markdown]
# ---
# ## 4 · Model Training & Initial Evaluation

# %%
# Retrieve default models dict
models = get_models()

# Train all models and evaluate
model_results = train_and_evaluate(X_train, X_test, y_train, y_test, models)

# Display a quick metrics table
results_summary = []
for name, metrics in model_results.items():
    results_summary.append({
        'Model': name,
        'Accuracy': metrics['accuracy'],
        'Macro F1-Score': metrics['f1_macro'],
        'Weighted F1-Score': metrics['f1_weighted']
    })
df_results = pd.DataFrame(results_summary).sort_values(by='Macro F1-Score', ascending=False)
df_results

# %% [markdown]
# ---
# ## 5 · Model Comparison Plots

# %%
# Reshape for plotting
df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Model', hue='Metric', data=df_melted, palette='muted')
plt.title('Support Ticket Category Classifiers Performance Comparison', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Metric Score (higher is better)', fontsize=12)
plt.ylabel('Model Name', fontsize=12)
plt.xlim(0.0, 1.05)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()

# Save performance comparison chart
comp_path = os.path.join(FIGDIR, 'category_model_comparison.png')
plt.savefig(comp_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Performance comparison chart saved to: {comp_path}")

# %% [markdown]
# ---
# ## 6 · Hyperparameter Tuning on Best Model
# Let's run a grid search to optimize hyperparameters (regularization strength C and ngram_range) 
# for the top performing baseline, which is usually **Logistic Regression** or **Linear SVC**.

# %%
# Tune Logistic Regression (vectorizer + classifier pipeline)
best_pipeline = tune_best_model(X_train_raw, y_train, tfidf_vectorizer)

# %% [markdown]
# ---
# ## 7 · Detailed Evaluation of the Best Model

# %%
# Evaluate the tuned pipeline on test data
y_pred = best_pipeline.predict(X_test_raw)
best_acc = accuracy_score(y_test, y_pred)
best_f1_macro = f1_score(y_test, y_pred, average='macro')

print("=" * 60)
print(f"Tuned Category Pipeline Test Performance:")
print(f"  Accuracy  : {best_acc:.2%}")
print(f"  Macro F1  : {best_f1_macro:.2%}")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Save report text
report_path = os.path.join(project_root, 'reports', 'classification_report.txt')
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("🎫 CUSTOMER SUPPORT TICKET CATEGORY CLASSIFICATION REPORT\n")
    f.write("========================================================\n\n")
    f.write(f"Best Model Pipeline: TF-IDF Vectorizer + Logistic Regression\n")
    f.write(f"Test Accuracy: {best_acc:.4f}\n")
    f.write(f"Test Macro F1: {best_f1_macro:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, target_names=class_names))
print(f"Report saved to: {report_path}")

# %%
# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Category Classifier Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Actual Category', fontsize=12)
plt.xlabel('Predicted Category', fontsize=12)
plt.tight_layout()

cm_path = os.path.join(FIGDIR, 'category_confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Confusion matrix saved to: {cm_path}")

# %%
# ── Analyze Top Keywords (Features Importance) per Class ──────────────────────
# Extract feature importance mapping from the best model classifier
clf = best_pipeline.named_steps['classifier']
vec = best_pipeline.named_steps['vectorizer']

top_features = get_top_features_per_class(vec, clf, class_names, n_top=10)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, (cat_name, features) in enumerate(top_features.items()):
    terms, scores = zip(*features)
    ax = axes[idx]
    sns.barplot(x=list(scores), y=list(terms), ax=ax, palette='Blues_r')
    ax.set_title(f"Top 10 Terms: {cat_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel('Coefficient Score')
    ax.set_ylabel('Term')

# Disable empty subplots if any (we have 5 categories and 6 slots)
if len(class_names) < len(axes):
    axes[-1].axis('off')
    
plt.suptitle('Logistic Regression Feature Coefficients by Class', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

feat_path = os.path.join(FIGDIR, 'category_feature_importance.png')
plt.savefig(feat_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Feature importance plot saved to: {feat_path}")

# %%
# ── Analysis of Misclassifications ──────────────────────────────────────────
df_test = pd.DataFrame({
    'text': X_test_raw,
    'actual': label_encoder.inverse_transform(y_test),
    'predicted': label_encoder.inverse_transform(y_pred)
})
misclassified = df_test[df_test['actual'] != df_test['predicted']]
print(f"\nTotal misclassified tickets in test set: {len(misclassified)} ({len(misclassified)/len(y_test):.2%})")

print("\nSample Misclassifications:")
print("-" * 80)
for idx, row in misclassified.head(5).iterrows():
    print(f"Actual: {row['actual']} | Predicted: {row['predicted']}")
    print(f"Text: \"{row['text'][:180]}...\"")
    print("-" * 80)

# %% [markdown]
# ---
# ## 8 · Model Serialization
# Save the final tuned pipeline and label encoder to disk for production usage.

# %%
cat_model_path = os.path.join(MODELS_DIR, 'category_classifier.pkl')
label_encoder_path = os.path.join(MODELS_DIR, 'category_label_encoder.pkl')

save_model(best_pipeline, cat_model_path)
save_model(label_encoder, label_encoder_path)

print("\n🎉 Category Classification modeling phase complete!")
