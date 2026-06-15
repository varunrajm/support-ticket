"""
data_generator.py
Generates 5,000 realistic synthetic customer support tickets.
Each ticket includes a subject, description, category, priority,
and various metadata columns.
"""
import random
import os
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)

# ── Ticket vocabulary and templates ──────────────────────────────────────────
CATEGORIES = [
    "Billing Issue",
    "Technical Issue",
    "Account Access",
    "Product Inquiry",
    "Feature Request",
]

CHANNELS = ["Email", "Chat", "Phone", "Social Media", "Web Form"]
PRODUCTS = [
    "Basic Plan",
    "Pro Plan",
    "Enterprise Plan",
    "Mobile Add-on",
    "Analytics Add-on",
]
GENDERS   = ["Male", "Female", "Non-binary", "Prefer not to say"]
STATUSES  = ["Open", "Pending", "Resolved", "Closed"]

# ── Realistic ticket templates by category ────────────────────────────────────
TEMPLATES = {
    "Billing Issue": {
        "subjects": [
            "Incorrect charge on my account",
            "Double billing issue",
            "Unexpected subscription fee",
            "Refund request for duplicate payment",
            "Invoice shows wrong amount",
            "Charged after cancellation",
            "Credit card billed twice this month",
            "Wrong plan amount on invoice",
        ],
        "bodies": [
            "Hi, I noticed my account was charged {amount} on {date} but I only signed up for the {plan} plan which costs {correct_amount}. Can you please look into this and process a refund?",
            "Hello support team, I was billed twice for my subscription this month — once on {date} and again two days later. I only authorized one payment. Please refund the duplicate charge of {amount}.",
            "I cancelled my subscription on {date} but my card was still charged {amount} the following week. This is unacceptable. I need a full refund immediately.",
            "My latest invoice shows {amount} but according to my plan details it should be {correct_amount}. There seems to be an error. Can someone from billing review this?",
            "I upgraded from Basic to Pro last month but I'm still being charged the Enterprise rate of {amount}. Please fix this billing error and adjust my invoice.",
            "I submitted a refund request last week (ticket #{ticket_num}) and haven't heard back. The overcharge amount was {amount}. Please process this urgently.",
        ],
        "high_priority": [
            "URGENT: My account has been charged {amount} without authorization. Please reverse this transaction immediately or I will dispute with my bank.",
            "Critical billing error — I've been charged {amount} three times this week! My card is maxed out because of this. Need immediate resolution.",
        ],
        "low_priority": [
            "Just a quick question — I noticed a small discrepancy of {amount} on my last invoice. No rush, but could someone take a look when you get a chance?",
            "I'm wondering if there's a discount available for annual billing. I'm currently on the monthly {plan} plan. When you have a moment, let me know.",
        ],
    },
    "Technical Issue": {
        "subjects": [
            "App crashes on startup",
            "Unable to load dashboard",
            "Error 500 when saving data",
            "Login page not loading",
            "API returning incorrect results",
            "Performance extremely slow",
            "File upload not working",
            "Export feature broken",
        ],
        "bodies": [
            "Hi there, I'm getting an error when I try to open the app on my {device}. It shows a white screen for a few seconds and then crashes. This started happening after yesterday's update. I've tried reinstalling but the issue persists.",
            "The dashboard has been completely unusable since this morning. It just shows a loading spinner forever and never loads. My team depends on this tool for our daily standup. This is blocking our work.",
            "Every time I click 'Save' on the report editor I get a '500 Internal Server Error'. I've tried different browsers and the same thing happens. Please fix this urgently.",
            "The API endpoint {endpoint} started returning 404 errors as of {date}. This is breaking our integration and affecting our customers. We need an ETA on a fix.",
            "Pages are loading extremely slowly — it's taking over 30 seconds to load the analytics page. This started after last Friday's deployment. Please investigate.",
            "The file upload feature is broken. I'm trying to upload a CSV file ({filesize}) and I keep getting 'Upload failed' after waiting several minutes. I've tried multiple files.",
        ],
        "high_priority": [
            "CRITICAL: Production system is completely down. All users in our organization are getting 503 errors. This is affecting live operations. Need immediate support.",
            "URGENT: Security vulnerability detected. Our logs show unauthorized API access attempts. Need your security team to investigate immediately.",
        ],
        "low_priority": [
            "Minor UI glitch — the sidebar menu sometimes appears slightly misaligned on wide screens. Not a blocker but thought you should know. Happy to provide a screenshot.",
            "The dark mode toggle doesn't remember my preference between sessions. I have to reset it every time I log in. Not urgent but a bit annoying.",
        ],
    },
    "Account Access": {
        "subjects": [
            "Cannot log in to my account",
            "Password reset email not received",
            "Account locked after failed attempts",
            "Two-factor authentication not working",
            "Forgot account email address",
            "Cannot access admin panel",
            "SSO login broken",
            "Account suspended unexpectedly",
        ],
        "bodies": [
            "I'm unable to log in to my account. I enter my email {email_placeholder} and password but it just says 'Invalid credentials'. I haven't changed my password recently.",
            "I requested a password reset email over an hour ago but haven't received anything. I've checked my spam folder. My account email is {email_placeholder}. Please resend.",
            "My account appears to be locked. I tried to log in a few times and now I'm seeing 'Account temporarily locked'. I need to access my account urgently for a client meeting.",
            "The two-factor authentication code I receive via SMS doesn't work. I enter the 6-digit code within the time window but it says 'Invalid code'. This has been happening since yesterday.",
            "I'm trying to log in using our company SSO but I'm getting a SAML authentication error. This is affecting my entire team — none of us can access the platform.",
            "I need to access the admin panel to add a new team member but the 'Admin' option doesn't appear in my account menu. I was previously an admin. Has something changed?",
        ],
        "high_priority": [
            "LOCKED OUT: I'm completely unable to access my account and have a client presentation in 2 hours that requires access to the platform. Please unlock my account immediately.",
            "URGENT: Our entire team has been locked out following an admin change. No one in the organization can log in. This is a critical situation affecting business operations.",
        ],
        "low_priority": [
            "I'd like to update my profile picture but can't find the option in settings. Could you point me to where that is? No rush at all.",
            "I want to change my account timezone but the setting doesn't seem to save. It's a minor thing but would be nice to fix.",
        ],
    },
    "Product Inquiry": {
        "subjects": [
            "Question about pricing plans",
            "Comparing Pro vs Enterprise features",
            "Integration with third-party tools",
            "Data export capabilities",
            "Team collaboration features",
            "API documentation question",
            "Trial extension request",
            "Volume discount inquiry",
        ],
        "bodies": [
            "Hi, I'm evaluating your product for my company of {team_size} people. Could you explain the main differences between the Pro and Enterprise plans? We're particularly interested in the user limits and collaboration features.",
            "We currently use {other_tool} for {use_case}. I'd like to know if your product integrates with it natively or if we'd need to use something like Zapier.",
            "Before we commit to an annual plan, I'd like to know about your data export capabilities. Can we export everything in CSV format? What about API access to our historical data?",
            "I'm on the free trial and it's almost expired. I haven't had a chance to fully test all the features yet. Is it possible to get a 2-week extension?",
            "We're a team of {team_size} and currently using the Pro plan. We've grown recently and I want to understand what we'd get by upgrading to Enterprise. Can we schedule a demo?",
            "Could you clarify whether the {feature} is included in the Basic plan or only in higher tiers? Your pricing page isn't entirely clear on this.",
        ],
        "high_priority": [
            "We have a board meeting tomorrow and need to demo your product's reporting features. Can someone from sales or support jump on a quick call with me in the next hour?",
            "We're making a final purchasing decision by end of day today. I need clarity on the SLA guarantees and uptime commitments for the Enterprise plan ASAP.",
        ],
        "low_priority": [
            "I was just curious — does your platform support right-to-left languages? We might expand internationally at some point. No immediate need.",
            "Wondering if you have a desktop app in addition to the web version? Would be nice to have. Just a question, no urgency.",
        ],
    },
    "Feature Request": {
        "subjects": [
            "Dark mode toggle suggestion",
            "Bulk export functionality request",
            "Calendar integration feature",
            "Custom notification settings",
            "Keyboard shortcut support",
            "Mobile app improvements",
            "Advanced filtering options",
            "Automated report scheduling",
        ],
        "bodies": [
            "I'd love to see a dark mode option in the interface. I work late at night and the current bright white theme is hard on the eyes during long sessions. Many of your competitors already offer this.",
            "It would be incredibly useful to have a bulk export feature where I can download all my data at once in a single ZIP file. Currently I have to export each section individually which is very time-consuming.",
            "A calendar integration (Google Calendar or Outlook) would make the task scheduling feature much more useful. Right now we have to manually copy dates back and forth.",
            "Could you add the ability to schedule automated weekly reports to be emailed to stakeholders? This would save our team hours each week of manual reporting.",
            "It would be great if the data tables supported more advanced filtering — for example, filtering by date range, multiple categories at once, or custom regex patterns.",
            "I think many users would benefit from keyboard shortcut support. Even basic ones like Ctrl+S to save, Ctrl+F to search, etc. It would speed up my workflow a lot.",
        ],
        "high_priority": [
            "This is a feature that's blocking our adoption of your product. We absolutely need the ability to add custom user roles with granular permissions. Without this we can't deploy to our team.",
            "We urgently need an audit log feature before we can sign the enterprise contract. Our compliance team requires this. Can you tell me if this is on the roadmap?",
        ],
        "low_priority": [
            "Not urgent at all, but it would be a nice touch if the app remembered which tab I was last on when I return to a page. Just a quality-of-life suggestion!",
            "A small suggestion — it would be cool to have a 'recently viewed' section on the dashboard. Nothing critical, just a UI enhancement idea.",
        ],
    },
}

# ── Priority assignment logic ─────────────────────────────────────────────────
PRIORITY_WEIGHTS = {
    "Billing Issue":     {"High": 0.25, "Medium": 0.50, "Low": 0.25},
    "Technical Issue":   {"High": 0.40, "Medium": 0.45, "Low": 0.15},
    "Account Access":    {"High": 0.45, "Medium": 0.40, "Low": 0.15},
    "Product Inquiry":   {"High": 0.10, "Medium": 0.30, "Low": 0.60},
    "Feature Request":   {"High": 0.05, "Medium": 0.20, "Low": 0.75},
}

# ── Helper variables for realistic template substitution ──────────────────────
AMOUNTS         = ["$9.99", "$19.99", "$49.00", "$99.00", "$149.99", "$299.00"]
CORRECT_AMOUNTS = ["$9.99", "$19.99", "$29.99", "$49.00"]
DATES           = ["June 1st", "June 5th", "June 10th", "May 28th", "last Tuesday", "two days ago"]
PLANS           = ["Basic", "Pro", "Enterprise", "Starter"]
TICKET_NUMS     = [str(random.randint(10000, 99999)) for _ in range(100)]
DEVICES         = ["iPhone 14", "Samsung Galaxy S23", "MacBook Pro", "Windows laptop", "iPad"]
ENDPOINTS       = ["/api/v1/users", "/api/v2/reports", "/api/v1/export", "/api/v2/analytics"]
FILESIZES       = ["2MB", "5MB", "15MB", "45MB"]
TEAM_SIZES      = ["10", "25", "50", "100", "200+"]
OTHER_TOOLS     = ["Salesforce", "HubSpot", "Slack", "Jira", "Notion", "Zapier"]
USE_CASES       = ["CRM", "project management", "customer communication", "data analysis"]
FEATURES        = ["API access", "custom reports", "team collaboration", "data export", "SSO integration"]


def _fill_template(template: str) -> str:
    """Replace placeholder tokens in a template string with random values."""
    replacements = {
        "{amount}":          random.choice(AMOUNTS),
        "{correct_amount}":  random.choice(CORRECT_AMOUNTS),
        "{date}":            random.choice(DATES),
        "{plan}":            random.choice(PLANS),
        "{ticket_num}":      random.choice(TICKET_NUMS),
        "{device}":          random.choice(DEVICES),
        "{endpoint}":        random.choice(ENDPOINTS),
        "{filesize}":        random.choice(FILESIZES),
        "{team_size}":       random.choice(TEAM_SIZES),
        "{other_tool}":      random.choice(OTHER_TOOLS),
        "{use_case}":        random.choice(USE_CASES),
        "{feature}":         random.choice(FEATURES),
        "{email_placeholder}": "my registered email",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _pick_body(category: str, priority: str) -> str:
    """Pick a body template weighted toward priority-appropriate templates."""
    templates_for_cat = TEMPLATES[category]
    if priority == "High" and random.random() < 0.45:
        pool = templates_for_cat.get("high_priority", templates_for_cat["bodies"])
    elif priority == "Low" and random.random() < 0.45:
        pool = templates_for_cat.get("low_priority", templates_for_cat["bodies"])
    else:
        pool = templates_for_cat["bodies"]
    return _fill_template(random.choice(pool))


def generate_dataset(num_samples: int = 5000) -> pd.DataFrame:
    print(f"Generating {num_samples} synthetic customer support tickets…")
    rows = []
    for _ in range(num_samples):
        category = random.choice(CATEGORIES)
        weights  = PRIORITY_WEIGHTS[category]
        priority = random.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]

        subject = random.choice(TEMPLATES[category]["subjects"])
        body    = _pick_body(category, priority)

        # Resolution time correlated with priority
        if priority == "High":
            resolution = f"{random.randint(1, 12)} hours"
        elif priority == "Medium":
            resolution = f"{random.randint(1, 3)} days"
        else:
            resolution = f"{random.randint(2, 7)} days"

        rows.append({
            "Ticket_Subject":              subject,
            "Ticket_Description":          body,
            "Ticket_Type":                 category,
            "Ticket_Priority":             priority,
            "Ticket_Channel":              random.choice(CHANNELS),
            "Product_Purchased":           random.choice(PRODUCTS),
            "Gender":                      random.choice(GENDERS),
            "Customer_Age":                random.randint(18, 68),
            "Customer_Satisfaction_Rating": random.randint(1, 5),
            "Time_to_Resolution":          resolution,
            "Ticket_Status":               random.choice(STATUSES),
        })

    df = pd.DataFrame(rows)
    out_dir  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "customer_support_tickets.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {num_samples:,} tickets → {out_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
