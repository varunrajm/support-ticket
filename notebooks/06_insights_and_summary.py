# %% [markdown]
# # 📈 06 — Business Insights & Summary
# **Customer Support Ticket Classification Project**
#
# This final notebook acts as a bridge between machine learning metrics and business value.
# We load our serialized models, run live inference on new unseen examples, model the return 
# on investment (ROI), outline the deployment architecture, and summarize the overall system.
#
# **Sections:**
# 1. Setup & Loading Models
# 2. Live Demo Inference
# 3. Operations Routing Simulation
# 4. Business Value & ROI Calculator
# 5. Production Deployment Blueprint
# 6. Final Project Wrap-Up

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

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Color palette definition
PALETTE = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
sns.set_palette(PALETTE)

FIGDIR = os.path.join(project_root, 'reports', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

from src.predict import load_models, classify_ticket

# %% [markdown]
# ---
# ## 1 · Load Serialized Models

# %%
try:
    cat_model, prio_model = load_models()
    print("✅ Models loaded successfully!")
    print(f"  Category Model: {type(cat_model).__name__}")
    print(f"  Priority Model: {type(prio_model).__name__}")
except Exception as e:
    print(f"❌ Failed to load models. Error: {e}")
    print("Ensure you have run Notebooks 04 and 05 first to generate the models.")

# %% [markdown]
# ---
# ## 2 · Live Demo Inference
# We test the trained pipeline on 5 realistic support ticket drafts that the model 
# has never seen before, including mixed casing, typos, and specific domain triggers.

# %%
test_tickets = [
    {
        'subject': 'Billing error: double charge on card',
        'description': 'Hello, I checked my bank account statement this morning and noticed that I was charged $49.00 twice for my laptop subscription on June 10th. I only purchased it once. Can you check this double charge and refund the extra $49.00 ASAP? Thank you.'
    },
    {
        'subject': 'Tablet app keeps crashing after start screen',
        'description': 'Hey support team, I downloaded the new version of the app yesterday on my android device. When I click to open it, the screen goes black for a second and then crashes immediately. This is not working at all and I have a presentation tomorrow. Urgent fix requested.'
    },
    {
        'subject': 'How do I add a new payment method?',
        'description': 'Hello, I would like to purchase the smartwatch. Does your payment system accept Apple Pay? Or do I need to enter a traditional credit card? I could not find this information on the checkout page.'
    },
    {
        'subject': 'Locked out of system / password link broken',
        'description': 'My account password has expired. I tried to use the automated password reset form, but the link sent to my email redirects to an error page. Now my account access is completely locked and I cannot sign in. Please unlock my account manually!'
    },
    {
        'subject': 'Request: Dark theme for reports page',
        'description': 'It would be really great if the reporting dashboard had a dark mode toggle option. The white background is very bright during night shifts, making it hard to look at the screen for long hours. Please consider adding this feature.'
    }
]

inference_results = []
for i, ticket in enumerate(test_tickets, 1):
    res = classify_ticket(ticket['subject'], ticket['description'], cat_model, prio_model)
    inference_results.append({
        'Ticket ID': f"DEMO-{i:02d}",
        'Subject': ticket['subject'],
        'Predicted Category': res['category'],
        'Category Conf': f"{res['category_confidence']:.1%}",
        'Predicted Priority': res['priority'],
        'Priority Conf': f"{res['priority_confidence']:.1%}",
        'Routing Decision': res['recommended_action'].split('|')[0].replace('ROUTE: ', '')
    })

df_demo_results = pd.DataFrame(inference_results)
df_demo_results

# %% [markdown]
# ---
# ## 3 · Operations Routing Simulation
# Let's inspect how our models would classify a larger batch of raw tickets and see 
# the distribution of recommended routing decisions.

# %%
# Load original dataset to simulate batch classification
RAW_DATA_PATH = os.path.join(project_root, 'data', 'raw', 'customer_support_tickets.csv')
if os.path.exists(RAW_DATA_PATH):
    df_raw = pd.read_csv(RAW_DATA_PATH).head(500)  # Simulate on first 500 tickets
    
    # Classify batch
    from src.predict import batch_classify
    df_classified = batch_classify(df_raw, cat_model, prio_model)
    
    # Count routing recommendations
    routing_counts = df_classified['Recommended_Routing_Action'].value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=routing_counts.values, y=routing_counts.index, palette='plasma')
    plt.title('Simulated Queue Distribution (SLA-based Routing)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Support Tickets', fontsize=12)
    plt.tight_layout()
    
    routing_plot_path = os.path.join(FIGDIR, 'simulated_routing_queues.png')
    plt.savefig(routing_plot_path, dpi=300)
    plt.show()
    print(f"Routing queue distribution plot saved to: {routing_plot_path}")
else:
    print("Raw dataset not found. Skipping simulation plot.")

# %% [markdown]
# ---
# ## 4 · Business Value & Return on Investment (ROI)
# Let's model the tangible savings in labor and SLA adherence this system brings to an organization.

# %%
# Business Parameters (Adjustable variables)
daily_ticket_volume = 1200          # Average tickets received per day
sorting_time_per_ticket_min = 3.0  # Time in minutes a human agent spends reading & categorizing
labor_cost_per_hour_usd = 22.0     # Hourly wage of support triage staff
system_accuracy_rate = 0.85        # Accuracy rate of ML classification (e.g. 85%)

# 1. Total hours spent triage manually per day
manual_hours_triage_daily = (daily_ticket_volume * sorting_time_per_ticket_min) / 60
manual_cost_daily = manual_hours_triage_daily * labor_cost_per_hour_usd

# 2. Savings with ML Assist
# The system auto-routes X% of tickets with high confidence directly to queues.
# Humans only audit or route the misclassified or low confidence ones (e.g. 15%).
hours_saved_daily = manual_hours_triage_daily * system_accuracy_rate
financial_savings_daily = hours_saved_daily * labor_cost_per_hour_usd
financial_savings_monthly = financial_savings_daily * 30.4
financial_savings_yearly = financial_savings_daily * 365

print("=" * 70)
print("             💰 SUPPORT AUTOMATION ROI ANALYSIS SUMMARY")
print("=" * 70)
print(f"Daily Ticket Ingest Volume   : {daily_ticket_volume:,} tickets / day")
print(f"Manual Triage Labor Required : {manual_hours_triage_daily:.1f} hours / day")
print(f"Daily Triage Labor Cost      : ${manual_cost_daily:,.2f} / day")
print(f"System Auto-Routing Accuracy : {system_accuracy_rate:.1%}")
print("-" * 70)
print(f"⏳ Operational Hours Saved   : {hours_saved_daily:.1f} hours / day")
print(f"💵 Daily Financial Savings   : ${financial_savings_daily:,.2f} / day")
print(f"💵 Monthly Financial Savings : ${financial_savings_monthly:,.2f} / month")
print(f"🚀 Annual Projected Savings  : ${financial_savings_yearly:,.2f} / year")
print("=" * 70)

# %% [markdown]
# ---
# ## 5 · Production Deployment Blueprint
# In a real production environment, this classifier is typically packaged as a REST API and integrated into the ticketing pipeline.
#
# ```
#   [ Incoming Ticket ] (Email, Chat, Portal)
#            │
#            ▼
# ┌───────────────────┐
# │   Zendesk / Jira  │ (Ticketing Platform webhook)
# └──────────┬────────┘
#            │ (POST request with raw text JSON payload)
#            ▼
# ┌───────────────────┐
# │    FastAPI App    │ (Running ML Classifier models)
# │ ── ── ── ── ── ── │
# │ - preprocess text │
# │ - run inference   │
# └──────────┬────────┘
#            │ (Returns predicted Category, Priority, and Routing Action)
#            ▼
# ┌───────────────────┐
# │   Zendesk / Jira  │ (Ticketing platform updates routing fields & tags)
# └──────────┬────────┘
#            ├──────────────────────┼──────────────────────┐
#            ▼                      ▼                      ▼
#   [ Billing Queue ]      [ Tech Ops Queue ]     [ Security Ops Queue ]
#     (SLA: 12 Hours)        (SLA: 8 Hours)         (SLA: 30 Minutes)
# ```

# %% [markdown]
# ---
# ## 6 · Final Project Wrap-Up
# We successfully designed and implemented an end-to-end Machine Learning support triage pipeline:
#
# 1. **Data Ingestion & Simulation**: Created a realistic dataset of 5,000 tickets reflecting SaaS scenarios.
# 2. **Text Preprocessing**: Implemented custom cleaner functions (regex, NLTK tokenizers, and lemmatizers).
# 3. **Feature Engineering**: Vectorized text with sublinear TF-IDF and engineered metadata features (uppercase ratios, exclamation marks).
# 4. **Multi-Output Modeling**: Trained separate estimators for Category (Accuracy: ~85%) and Priority (Accuracy: ~78%).
# 5. **Triage Automation**: Packaged the system as an inference engine (`src/predict.py`) that returns actionable business decisions and routing rules.
#
# By deploying this system, companies can save **thousands of triage hours annually**, optimize SLA compliance, and route critical bugs or customer account lockouts to specialist teams within minutes.
