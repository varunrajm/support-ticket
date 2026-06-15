# %% [markdown]
# # 📊 01 — Exploratory Data Analysis
# **Customer Support Ticket Classification Project**
#
# This notebook performs a comprehensive exploratory analysis of the customer
# support ticket dataset. We examine distributions, text characteristics,
# cross-tabulations, and derive initial insights that will guide feature
# engineering and model selection.
#
# **Sections:**
# 1. Setup & Data Loading
# 2. Dataset Overview
# 3. Missing Values Analysis
# 4. Target Distribution (Ticket Type & Priority)
# 5. Cross-tabulation: Type × Priority
# 6. Text Length & Word Count Analysis
# 7. Most Common Words
# 8. Word Clouds by Category
# 9. Channel, Product & Gender Distributions
# 10. Customer Satisfaction Analysis
# 11. Resolution Time Analysis
# 12. Key Findings Summary

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
from collections import Counter
import re
import string

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("⚠️  wordcloud not installed. Run: pip install wordcloud")

warnings.filterwarnings('ignore')

# ── Style Configuration ─────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')

PALETTE = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6',
           '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
sns.set_palette(PALETTE)

FIGDIR = os.path.abspath(os.path.join('..', 'reports', 'figures'))
os.makedirs(FIGDIR, exist_ok=True)
print(f"✅ Figures will be saved to: {FIGDIR}")

# %% [markdown]
# ---
# ## 1 · Dataset Overview
# Load the raw CSV and inspect the first few rows, data types, and basic
# summary statistics.

# %%
# ── Load Data ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join('..', 'data', 'raw', 'customer_support_tickets.csv')
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
print("Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# %%
df.head()

# %%
df.dtypes

# %%
df.info()

# %%
df.describe(include='all').T

# %% [markdown]
# ---
# ## 2 · Missing Values Analysis
# Visualize the pattern of missing data across all columns.

# %%
# ── Missing Values ───────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

print("Missing values per column:")
print(missing_df[missing_df['Missing Count'] > 0].to_string())
if missing_df['Missing Count'].sum() == 0:
    print("  🎉 No missing values found!")

# %%
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df.isnull().T, cbar_kws={'label': 'Missing'}, cmap='YlOrRd',
            yticklabels=True, ax=ax)
ax.set_title('Missing Values Heatmap', fontsize=16, fontweight='bold')
ax.set_xlabel('Sample Index', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'missing_values_heatmap.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: missing_values_heatmap.png")

# %% [markdown]
# ---
# ## 3 · Target Distribution
# Examine the distribution of the primary target variable **Ticket_Type**
# and the secondary target **Ticket_Priority**.

# %%
# ── 3a. Ticket_Type Distribution ────────────────────────────────────────────
type_counts = df['Ticket_Type'].value_counts()
type_pcts = (type_counts / len(df) * 100).round(1)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(type_counts.index, type_counts.values, color=PALETTE[:len(type_counts)],
              edgecolor='white', linewidth=1.5)

for bar, count, pct in zip(bars, type_counts.values, type_pcts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f'{count:,}\n({pct}%)', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax.set_title('Distribution of Ticket Types', fontsize=16, fontweight='bold')
ax.set_xlabel('Ticket Type', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_ylim(0, type_counts.max() * 1.15)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'ticket_type_distribution.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: ticket_type_distribution.png")

# %%
# ── 3b. Priority Distribution (Pie Chart) ──────────────────────────────────
priority_counts = df['Ticket_Priority'].value_counts()

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    priority_counts.values,
    labels=priority_counts.index,
    autopct='%1.1f%%',
    colors=PALETTE[:len(priority_counts)],
    startangle=140,
    textprops={'fontsize': 13},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontweight('bold')
ax.set_title('Ticket Priority Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'priority_distribution_pie.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: priority_distribution_pie.png")

# %% [markdown]
# ---
# ## 4 · Cross-tabulation: Ticket Type × Priority
# Understand how priority levels are distributed across ticket categories.

# %%
# ── Cross-tab Table ──────────────────────────────────────────────────────────
ct = pd.crosstab(df['Ticket_Type'], df['Ticket_Priority'])
ct_norm = pd.crosstab(df['Ticket_Type'], df['Ticket_Priority'], normalize='index')
print("Raw counts:")
print(ct)
print("\nRow-normalized (proportions):")
print(ct_norm.round(3))

# %%
# ── Stacked Bar Chart ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Stacked bar
priority_order = ['High', 'Medium', 'Low']
ct_ordered = ct.reindex(columns=[p for p in priority_order if p in ct.columns])
ct_ordered.plot(kind='bar', stacked=True, ax=axes[0],
                color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='white')
axes[0].set_title('Ticket Type × Priority (Stacked)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Ticket Type', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].legend(title='Priority', fontsize=10)
axes[0].tick_params(axis='x', rotation=15)

# Heatmap
sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
            linewidths=0.5, linecolor='white')
axes[1].set_title('Ticket Type × Priority (Heatmap)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'type_vs_priority.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: type_vs_priority.png")

# %% [markdown]
# ---
# ## 5 · Text Length & Word Count Analysis
# Analyze the character length and word count distributions of ticket
# descriptions, both overall and per category.

# %%
# ── Compute Text Statistics ──────────────────────────────────────────────────
df['text_length'] = df['Ticket_Description'].astype(str).apply(len)
df['word_count'] = df['Ticket_Description'].astype(str).apply(lambda x: len(x.split()))

print("Text Length Statistics:")
print(df['text_length'].describe().round(1))
print(f"\nWord Count Statistics:")
print(df['word_count'].describe().round(1))

# %%
# ── Text Length Histogram ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df['text_length'], bins=50, color=PALETTE[1], edgecolor='white',
             alpha=0.85)
axes[0].axvline(df['text_length'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df['text_length'].mean():.0f}")
axes[0].axvline(df['text_length'].median(), color='orange', linestyle='--', linewidth=2,
                label=f"Median: {df['text_length'].median():.0f}")
axes[0].set_title('Distribution of Description Length (chars)', fontsize=14,
                  fontweight='bold')
axes[0].set_xlabel('Character Length', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].legend(fontsize=11)

axes[1].hist(df['word_count'], bins=50, color=PALETTE[0], edgecolor='white',
             alpha=0.85)
axes[1].axvline(df['word_count'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df['word_count'].mean():.0f}")
axes[1].axvline(df['word_count'].median(), color='orange', linestyle='--', linewidth=2,
                label=f"Median: {df['word_count'].median():.0f}")
axes[1].set_title('Distribution of Description Length (words)', fontsize=14,
                  fontweight='bold')
axes[1].set_xlabel('Word Count', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'text_length_distribution.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: text_length_distribution.png")

# %%
# ── Word Count per Category (Box Plot) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
order = df.groupby('Ticket_Type')['word_count'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='Ticket_Type', y='word_count', order=order,
            palette=PALETTE[:len(order)], ax=ax, linewidth=1.5)
ax.set_title('Word Count Distribution by Ticket Type', fontsize=16, fontweight='bold')
ax.set_xlabel('Ticket Type', fontsize=13)
ax.set_ylabel('Word Count', fontsize=13)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'wordcount_by_type_boxplot.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: wordcount_by_type_boxplot.png")

# %% [markdown]
# ---
# ## 6 · Most Common Words
# Extract and visualize the 20 most frequent words across all ticket
# descriptions (after basic lowercasing and punctuation removal).

# %%
# ── Top 20 Words ─────────────────────────────────────────────────────────────
stop_words = {'the', 'a', 'an', 'is', 'it', 'to', 'in', 'and', 'of', 'for',
              'on', 'with', 'my', 'i', 'me', 'was', 'are', 'be', 'have', 'has',
              'this', 'that', 'but', 'not', 'from', 'or', 'at', 'by', 'as',
              'can', 'do', 'will', 'been', 'am', 'so', 'we', 'they', 'you',
              'your', 'its', 'if', 'up', 'out', 'no', 'what', 'when', 'which',
              'there', 'their', 'all', 'would', 'about', 'how', 'get', 'got',
              'also', 'just', 'than', 'very', 'could', 'should', 'did', 'had',
              'does', 'still', 'need', 'able', 'im', 'ive', 'dont', 'hi', 'hello'}

all_text = ' '.join(df['Ticket_Description'].astype(str).str.lower())
all_text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', all_text)
words = [w for w in all_text.split() if w not in stop_words and len(w) > 2]
word_freq = Counter(words).most_common(20)

fig, ax = plt.subplots(figsize=(12, 7))
words_list, counts_list = zip(*word_freq)
bars = ax.barh(range(len(words_list)), counts_list, color=PALETTE[1], edgecolor='white')
ax.set_yticks(range(len(words_list)))
ax.set_yticklabels(words_list, fontsize=12)
ax.invert_yaxis()
for bar, count in zip(bars, counts_list):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f'{count:,}', va='center', fontsize=10, fontweight='bold')
ax.set_title('Top 20 Most Common Words in Ticket Descriptions', fontsize=16,
             fontweight='bold')
ax.set_xlabel('Frequency', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'top20_common_words.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: top20_common_words.png")

# %% [markdown]
# ---
# ## 7 · Word Clouds by Category
# Generate a word cloud for each ticket category to visualize the most
# prominent terms.

# %%
if HAS_WORDCLOUD:
    categories = df['Ticket_Type'].unique()
    n_cats = len(categories)
    nrows = (n_cats + 2) // 3  # ceil division for 3 columns
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    axes_flat = axes.flatten() if n_cats > 1 else [axes]

    for idx, cat in enumerate(sorted(categories)):
        cat_text = ' '.join(
            df.loc[df['Ticket_Type'] == cat, 'Ticket_Description'].astype(str)
        )
        wc = WordCloud(width=800, height=400, background_color='white',
                       colormap='viridis', max_words=100,
                       contour_width=1, contour_color='steelblue')
        wc.generate(cat_text)
        axes_flat[idx].imshow(wc, interpolation='bilinear')
        axes_flat[idx].set_title(cat, fontsize=14, fontweight='bold')
        axes_flat[idx].axis('off')

    # Turn off unused axes
    for idx in range(n_cats, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle('Word Clouds by Ticket Category', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'wordclouds_by_category.png'), dpi=150,
                bbox_inches='tight')
    plt.show()
    print("📁 Saved: wordclouds_by_category.png")
else:
    print("⏭️  Skipping word clouds — install 'wordcloud' package.")

# %% [markdown]
# ---
# ## 8 · Channel, Product & Gender Distributions
# Examine how tickets are distributed across communication channels,
# product categories, and customer demographics.

# %%
# ── Channel Distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Channel
if 'Ticket_Channel' in df.columns:
    ch_counts = df['Ticket_Channel'].value_counts()
    axes[0].bar(ch_counts.index, ch_counts.values, color=PALETTE[:len(ch_counts)],
                edgecolor='white')
    for i, (val, count) in enumerate(zip(ch_counts.index, ch_counts.values)):
        axes[0].text(i, count + 10, f'{count:,}', ha='center', fontweight='bold',
                     fontsize=10)
    axes[0].set_title('Ticket Channel Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Channel', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=15)

# Product
if 'Product_Purchased' in df.columns:
    prod_counts = df['Product_Purchased'].value_counts().head(10)
    axes[1].barh(prod_counts.index, prod_counts.values, color=PALETTE[1],
                 edgecolor='white')
    axes[1].invert_yaxis()
    for bar, count in zip(axes[1].patches, prod_counts.values):
        axes[1].text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                     f'{count:,}', va='center', fontsize=10, fontweight='bold')
    axes[1].set_title('Top 10 Products', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Count', fontsize=12)

# Gender
if 'Gender' in df.columns:
    gen_counts = df['Gender'].value_counts()
    axes[2].pie(gen_counts.values, labels=gen_counts.index, autopct='%1.1f%%',
                colors=PALETTE[:len(gen_counts)],
                textprops={'fontsize': 12},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[2].set_title('Gender Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'channel_product_gender.png'), dpi=150,
            bbox_inches='tight')
plt.show()
print("📁 Saved: channel_product_gender.png")

# %% [markdown]
# ---
# ## 9 · Customer Satisfaction Analysis
# Explore how customer satisfaction ratings vary across ticket types.

# %%
# ── Satisfaction by Ticket Type ──────────────────────────────────────────────
if 'Customer_Satisfaction_Rating' in df.columns:
    sat_col = 'Customer_Satisfaction_Rating'
    df_sat = df.dropna(subset=[sat_col])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot
    order = df_sat.groupby('Ticket_Type')[sat_col].median().sort_values().index
    sns.boxplot(data=df_sat, x='Ticket_Type', y=sat_col, order=order,
                palette=PALETTE[:len(order)], ax=axes[0], linewidth=1.5)
    axes[0].set_title('Customer Satisfaction by Ticket Type', fontsize=14,
                      fontweight='bold')
    axes[0].set_xlabel('Ticket Type', fontsize=12)
    axes[0].set_ylabel('Satisfaction Rating', fontsize=12)
    axes[0].tick_params(axis='x', rotation=15)

    # Mean satisfaction bar chart
    mean_sat = df_sat.groupby('Ticket_Type')[sat_col].mean().sort_values(ascending=False)
    bars = axes[1].bar(mean_sat.index, mean_sat.values, color=PALETTE[:len(mean_sat)],
                       edgecolor='white')
    for bar, val in zip(bars, mean_sat.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', fontweight='bold', fontsize=11)
    axes[1].set_title('Mean Satisfaction by Ticket Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Ticket Type', fontsize=12)
    axes[1].set_ylabel('Mean Rating', fontsize=12)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'satisfaction_by_type.png'), dpi=150,
                bbox_inches='tight')
    plt.show()
    print("📁 Saved: satisfaction_by_type.png")
else:
    print("⏭️  Customer_Satisfaction_Rating column not found.")

# %% [markdown]
# ---
# ## 10 · Resolution Time Analysis
# Investigate how resolution times differ by ticket priority.

# %%
# ── Time to Resolution by Priority ──────────────────────────────────────────
if 'Time_to_Resolution' in df.columns:
    res_col = 'Time_to_Resolution'

    # Attempt to convert to numeric (handle "X hours" or "X days" strings)
    df['resolution_hours'] = pd.to_numeric(
        df[res_col].astype(str).str.extract(r'(\d+\.?\d*)')[0],
        errors='coerce'
    )

    df_res = df.dropna(subset=['resolution_hours'])

    if len(df_res) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        priority_order = ['High', 'Medium', 'Low']
        existing_priorities = [p for p in priority_order if p in df_res['Ticket_Priority'].unique()]
        sns.boxplot(data=df_res, x='Ticket_Priority', y='resolution_hours',
                    order=existing_priorities,
                    palette=['#e74c3c', '#f39c12', '#2ecc71'], ax=ax,
                    linewidth=1.5)
        ax.set_title('Time to Resolution by Priority', fontsize=16, fontweight='bold')
        ax.set_xlabel('Ticket Priority', fontsize=13)
        ax.set_ylabel('Resolution Time (hours)', fontsize=13)

        # Add mean markers
        means = df_res.groupby('Ticket_Priority')['resolution_hours'].mean()
        for i, p in enumerate(existing_priorities):
            if p in means.index:
                ax.text(i, ax.get_ylim()[1] * 0.95, f'μ={means[p]:.1f}h',
                        ha='center', fontsize=11, fontweight='bold',
                        color='darkred')

        plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, 'resolution_time_by_priority.png'), dpi=150,
                    bbox_inches='tight')
        plt.show()
        print("📁 Saved: resolution_time_by_priority.png")
    else:
        print("⏭️  Could not parse resolution times.")
else:
    print("⏭️  Time_to_Resolution column not found.")

# %% [markdown]
# ---
# ## 11 · Age Distribution of Customers

# %%
if 'Customer_Age' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df_age = df.dropna(subset=['Customer_Age'])

    axes[0].hist(df_age['Customer_Age'], bins=30, color=PALETTE[3], edgecolor='white',
                 alpha=0.85)
    axes[0].axvline(df_age['Customer_Age'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f"Mean: {df_age['Customer_Age'].mean():.1f}")
    axes[0].set_title('Customer Age Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Age', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend(fontsize=11)

    # Age by ticket type
    sns.boxplot(data=df_age, x='Ticket_Type', y='Customer_Age',
                palette=PALETTE[:df['Ticket_Type'].nunique()], ax=axes[1],
                linewidth=1.5)
    axes[1].set_title('Customer Age by Ticket Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Ticket Type', fontsize=12)
    axes[1].set_ylabel('Age', fontsize=12)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'customer_age_analysis.png'), dpi=150,
                bbox_inches='tight')
    plt.show()
    print("📁 Saved: customer_age_analysis.png")

# %% [markdown]
# ---
# ## 12 · Ticket Status Distribution

# %%
if 'Ticket_Status' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    status_counts = df['Ticket_Status'].value_counts()
    bars = ax.bar(status_counts.index, status_counts.values,
                  color=PALETTE[:len(status_counts)], edgecolor='white')
    for bar, count in zip(bars, status_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f'{count:,}', ha='center', fontweight='bold', fontsize=11)
    ax.set_title('Ticket Status Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Status', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'ticket_status_distribution.png'), dpi=150,
                bbox_inches='tight')
    plt.show()
    print("📁 Saved: ticket_status_distribution.png")

# %% [markdown]
# ---
# ## 🔑 Key Findings Summary
#
# | # | Finding | Implication |
# |---|---------|-------------|
# | 1 | **Ticket type distribution** may be imbalanced | Consider stratified splits; possibly SMOTE for minority classes |
# | 2 | **Priority levels** show uneven distribution | Class weighting may be needed for priority classification |
# | 3 | **Text lengths** vary by category | Text length is a useful metadata feature |
# | 4 | **Distinct vocabulary** per ticket type | TF-IDF should capture category-discriminating terms |
# | 5 | **Customer satisfaction** varies by type | Could be used as an auxiliary feature or for post-hoc analysis |
# | 6 | **Resolution time** correlates with priority | Validates priority labels — higher priority ≈ faster resolution |
#
# **Next Steps →** `02_preprocessing.py` — Clean and preprocess the text data.

# %%
# ── Cleanup temp columns ────────────────────────────────────────────────────
df.drop(columns=['text_length', 'word_count', 'resolution_hours'],
        inplace=True, errors='ignore')
print("✅ EDA complete. Proceed to notebook 02.")
