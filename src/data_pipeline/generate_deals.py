"""
SalesIntel AI — Synthetic CRM Deal Data Generator
===================================================

WHY SYNTHETIC DATA?
- We need realistic sales/CRM data to train ML models
- Real CRM data is proprietary and confidential — can't put it on GitHub
- By generating data ourselves, we control the patterns and correlations
- This means we KNOW what the model should learn, making debugging easier

HOW THIS WORKS:
- We generate ~5000 deals that mimic a real B2B sales pipeline
- Each deal has features that a real CRM (Salesforce, SAP CRM, HubSpot) would track
- We bake in REALISTIC correlations (not random noise), so our ML models
  can actually learn meaningful patterns later

REALISTIC PATTERNS WE EMBED:
1. Deals in later stages have higher win rates (obvious but important)
2. More sales activities (calls, emails) → higher win probability
3. Larger deals take longer to close and have lower win rates
4. Some industries convert better than others
5. Deals with competitors involved are harder to win
6. Deals stuck too long in the pipeline tend to die
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================
# CONFIGURATION
# ============================================================
# WHY separate config? In production ML systems, you never hardcode values.
# Configs make it easy to regenerate data with different parameters
# (e.g., "give me 50K rows" or "add a new industry").

NUM_DEALS = 5000  # Enough rows for ML training. Rule of thumb: 1000+ for tabular ML.

# WHY these industries? They represent typical B2B enterprise sales verticals.
# Different industries have different buying behaviors — this becomes a
# useful CATEGORICAL FEATURE for the model.
INDUSTRIES = [
    "Technology",       # Fast decision makers, higher win rate
    "Manufacturing",    # Slow, large deals
    "Financial Services",  # Compliance-heavy, medium speed
    "Healthcare",       # Budget cycles matter, medium win rate
    "Retail",           # Price-sensitive, lower deal sizes
    "Energy",           # Large deals, long cycles
    "Education",        # Budget-constrained, seasonal
    "Telecommunications",  # Medium deals, competitive market
]

# WHY these stages? This is a standard B2B sales funnel.
# The ORDER matters — it represents progression toward closing.
# We'll use the stage INDEX as a numeric feature later (ordinal encoding).
DEAL_STAGES = [
    "Prospecting",     # 0 - Just identified, very early
    "Qualification",   # 1 - Confirmed there's a real need
    "Proposal",        # 2 - Sent a proposal/quote
    "Negotiation",     # 3 - Discussing terms, pricing
    "Closed Won",      # 4 - Deal won!
    "Closed Lost",     # 5 - Deal lost
]

# WHY lead sources? Knowing WHERE a lead came from helps predict quality.
# Referrals typically convert better than cold outbound — a real pattern.
LEAD_SOURCES = [
    "Inbound Website",    # Came to us — medium quality
    "Outbound Cold Call",  # We reached out — lower quality
    "Referral",           # Someone recommended us — highest quality
    "Trade Show",         # Met at an event — medium quality
    "Partner",            # Channel partner — good quality
    "Social Media",       # LinkedIn etc — lower quality
]

SALES_REPS = [
    "Alice Johnson", "Bob Martinez", "Carol Williams", "David Chen",
    "Eva Kowalski", "Frank Okafor", "Grace Liu", "Henry Patel",
    "Iris Nakamura", "James Brown"
]

COMPANY_PREFIXES = [
    "Apex", "Nova", "Vertex", "Pinnacle", "Summit", "Atlas", "Zenith",
    "Catalyst", "Meridian", "Quantum", "Horizon", "Nexus", "Vanguard",
    "Cobalt", "Titan", "Forge", "Prism", "Axiom", "Orion", "Eclipse"
]

COMPANY_SUFFIXES = [
    "Technologies", "Solutions", "Industries", "Corp", "Systems",
    "Group", "Enterprises", "Global", "Partners", "Analytics",
    "Dynamics", "Networks", "Labs", "Innovations", "Services"
]


def generate_company_names(n: int) -> list[str]:
    """
    Generate realistic-sounding company names.
    WHY? Real CRM data has account names. Using realistic names makes
    the demo look professional, not like toy data.
    """
    rng = np.random.default_rng(42)
    names = set()
    while len(names) < n:
        prefix = rng.choice(COMPANY_PREFIXES)
        suffix = rng.choice(COMPANY_SUFFIXES)
        names.add(f"{prefix} {suffix}")
    return list(names)[:n]


def generate_deals(num_deals: int = NUM_DEALS, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic CRM deal data with realistic correlations.

    WHY seed=42?
    - Reproducibility. Anyone cloning this repo gets the EXACT same dataset.
    - This matters for ML — if your results aren't reproducible, they're not trustworthy.
    - 42 is a convention (Hitchhiker's Guide reference), any fixed number works.

    Returns:
        pd.DataFrame with columns matching a real CRM export
    """
    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------
    # STEP 1: Generate base fields (independent features)
    # ----------------------------------------------------------
    # These are features that don't depend on each other.
    # We generate them first, then build correlated features on top.

    industries = rng.choice(INDUSTRIES, size=num_deals)
    lead_sources = rng.choice(LEAD_SOURCES, size=num_deals)
    sales_reps = rng.choice(SALES_REPS, size=num_deals)
    company_names = rng.choice(generate_company_names(300), size=num_deals)

    # WHY different deal sizes per industry?
    # In reality, Manufacturing/Energy deals are much larger than Retail/Education.
    # This correlation is something the model should learn.
    industry_deal_size = {
        "Technology": (30_000, 80_000),
        "Manufacturing": (80_000, 250_000),
        "Financial Services": (50_000, 150_000),
        "Healthcare": (40_000, 120_000),
        "Retail": (15_000, 50_000),
        "Energy": (100_000, 300_000),
        "Education": (10_000, 40_000),
        "Telecommunications": (40_000, 100_000),
    }

    deal_amounts = []
    for ind in industries:
        low, high = industry_deal_size[ind]
        # WHY lognormal? Deal sizes in real life are right-skewed:
        # lots of small deals, few very large ones. Lognormal captures this.
        mean = (low + high) / 2
        amount = rng.lognormal(mean=np.log(mean), sigma=0.4)
        amount = np.clip(amount, low * 0.5, high * 2)  # Keep within reasonable bounds
        deal_amounts.append(round(amount, 2))

    deal_amounts = np.array(deal_amounts)

    # ----------------------------------------------------------
    # STEP 2: Generate time-based fields
    # ----------------------------------------------------------
    # WHY 18 months of data? Enough history for time-series forecasting later,
    # but not so much that patterns become stale.

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 6, 30)
    date_range_days = (end_date - start_date).days

    created_dates = [
        start_date + timedelta(days=int(rng.integers(0, date_range_days)))
        for _ in range(num_deals)
    ]

    # WHY link days_in_pipeline to deal_amount?
    # Bigger deals take longer to close — this is a well-known sales pattern.
    # A $10K deal might close in 2 weeks, a $200K deal takes months.
    amount_normalized = (deal_amounts - deal_amounts.min()) / (deal_amounts.max() - deal_amounts.min())
    base_days = 14 + (amount_normalized * 120)  # 14 to 134 days base
    noise = rng.normal(0, 15, size=num_deals)  # Add some randomness
    days_in_pipeline = np.clip(base_days + noise, 7, 300).astype(int)

    expected_close_dates = [
        created + timedelta(days=int(days))
        for created, days in zip(created_dates, days_in_pipeline)
    ]

    # ----------------------------------------------------------
    # STEP 3: Generate behavioral/activity fields
    # ----------------------------------------------------------
    # WHY track activities? In real CRM systems, the NUMBER of interactions
    # (emails, calls, meetings) is one of the strongest predictors of deal success.
    # More engagement = more likely to close.

    num_emails = rng.poisson(lam=8, size=num_deals)       # Average 8 emails per deal
    num_calls = rng.poisson(lam=4, size=num_deals)        # Average 4 calls
    num_meetings = rng.poisson(lam=2, size=num_deals)     # Average 2 meetings
    total_activities = num_emails + num_calls + num_meetings

    # WHY binary competitor flag?
    # Deals with competitors are harder to win. Simple but powerful feature.
    # ~40% of deals have a known competitor — realistic for B2B.
    competitor_involved = rng.binomial(1, 0.4, size=num_deals)

    # Number of stakeholders/contacts on the deal
    # More contacts usually means larger, more complex deals
    num_contacts = 1 + rng.poisson(lam=2, size=num_deals)

    # ----------------------------------------------------------
    # STEP 4: Determine deal outcome (THE TARGET VARIABLE)
    # ----------------------------------------------------------
    # THIS IS THE MOST IMPORTANT PART.
    #
    # WHY not just random 50/50 win/loss?
    # Because then the model learns nothing. We need realistic SIGNALS
    # so the model can learn which features predict winning.
    #
    # We compute a "win score" based on multiple factors, then convert
    # it to a probability. This mimics how real deals work:
    # many factors contribute to whether a deal closes.

    # Lead source quality — referrals convert better
    lead_source_boost = {
        "Inbound Website": 0.0,
        "Outbound Cold Call": -0.15,
        "Referral": 0.2,
        "Trade Show": 0.05,
        "Partner": 0.1,
        "Social Media": -0.1,
    }

    # Industry conversion tendencies
    industry_boost = {
        "Technology": 0.1,
        "Manufacturing": 0.0,
        "Financial Services": 0.05,
        "Healthcare": -0.05,
        "Retail": -0.1,
        "Energy": 0.05,
        "Education": -0.1,
        "Telecommunications": 0.0,
    }

    win_scores = np.zeros(num_deals)

    for i in range(num_deals):
        score = 0.0

        # Factor 1: More activities → higher win chance
        # WHY? Engaged prospects buy. This is the #1 predictor in real CRM analytics.
        activity_score = np.clip(total_activities[i] / 25.0, 0, 1) * 0.8
        score += activity_score

        # Factor 2: Deal amount (larger = harder to close)
        # WHY? Bigger deals have more decision-makers, longer approval chains.
        amount_penalty = -0.5 * amount_normalized[i]
        score += amount_penalty

        # Factor 3: Days in pipeline (too long = deal is dying)
        # WHY? "Time kills all deals" — real sales wisdom.
        if days_in_pipeline[i] > 120:
            score -= 0.6
        elif days_in_pipeline[i] > 90:
            score -= 0.3

        # Factor 4: Lead source quality
        score += lead_source_boost[lead_sources[i]] * 3

        # Factor 5: Industry
        score += industry_boost[industries[i]] * 3

        # Factor 6: Competitor makes it harder
        if competitor_involved[i]:
            score -= 0.5

        # Factor 7: More contacts can go either way
        # Many contacts = complex deal, but also = more buy-in
        if num_contacts[i] > 5:
            score -= 0.15  # Too many cooks
        elif num_contacts[i] >= 3:
            score += 0.15  # Good multi-threading

        win_scores[i] = score

    # Convert scores to probabilities using sigmoid
    # WHY sigmoid? It maps any score to a 0-1 probability range.
    # This is the same function used in logistic regression.
    # Base rate of ~45% win rate (realistic for B2B sales).
    # We scale scores so probabilities spread from ~10% to ~90% (not a narrow band).
    win_probabilities = 1 / (1 + np.exp(-(win_scores + 0.0)))

    # Determine actual outcomes based on probabilities
    outcomes_binary = rng.binomial(1, win_probabilities)

    # ----------------------------------------------------------
    # STEP 5: Assign deal stages based on outcome
    # ----------------------------------------------------------
    # WHY? If a deal is "Won", its stage should be "Closed Won".
    # For open deals, stage should correlate with how far along it is.

    stages = []
    final_outcomes = []

    for i in range(num_deals):
        if outcomes_binary[i] == 1:
            # 80% of won deals are Closed Won, 20% still in late stages
            if rng.random() < 0.8:
                stages.append("Closed Won")
                final_outcomes.append("Won")
            else:
                stages.append(rng.choice(["Negotiation", "Proposal"]))
                final_outcomes.append("Open")
        else:
            # 70% of lost deals are Closed Lost, 30% still in pipeline
            if rng.random() < 0.7:
                stages.append("Closed Lost")
                final_outcomes.append("Lost")
            else:
                stages.append(rng.choice(["Prospecting", "Qualification", "Proposal"]))
                final_outcomes.append("Open")

    # ----------------------------------------------------------
    # STEP 6: Assemble the DataFrame
    # ----------------------------------------------------------
    # WHY a DataFrame? It's the standard format for tabular ML in Python.
    # pandas DataFrames integrate directly with scikit-learn, XGBoost, etc.

    df = pd.DataFrame({
        "deal_id": [f"D-{10000 + i}" for i in range(num_deals)],
        "account_name": company_names,
        "industry": industries,
        "deal_amount": deal_amounts,
        "deal_stage": stages,
        "lead_source": lead_sources,
        "sales_rep": sales_reps,
        "created_date": created_dates,
        "expected_close_date": expected_close_dates,
        "days_in_pipeline": days_in_pipeline,
        "num_emails": num_emails,
        "num_calls": num_calls,
        "num_meetings": num_meetings,
        "total_activities": total_activities,
        "num_contacts": num_contacts,
        "competitor_involved": competitor_involved,
        "win_probability": np.round(win_probabilities, 3),
        "outcome": final_outcomes,
    })

    return df


def generate_revenue_timeseries(deals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate monthly revenue time series from closed-won deals.

    WHY a separate time series?
    - The deal-level data is for CLASSIFICATION (predict win/loss)
    - This time series is for FORECASTING (predict future revenue)
    - Two different ML problems from the same underlying data
    - Prophet/LSTM models need time-series format: date + value
    """
    won_deals = deals_df[deals_df["outcome"] == "Won"].copy()
    won_deals["close_month"] = pd.to_datetime(won_deals["expected_close_date"]).dt.to_period("M")

    monthly_revenue = (
        won_deals.groupby("close_month")["deal_amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "revenue", "count": "deals_closed"})
        .reset_index()
    )
    monthly_revenue["close_month"] = monthly_revenue["close_month"].dt.to_timestamp()
    monthly_revenue = monthly_revenue.rename(columns={"close_month": "date"})

    return monthly_revenue


# ============================================================
# MAIN — Run this script to generate and save all datasets
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SalesIntel AI - Generating Synthetic CRM Data")
    print("=" * 60)

    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    raw_data_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    # Generate deals
    print("\n[1/3] Generating deal data...")
    deals = generate_deals()
    deals_path = os.path.join(raw_data_dir, "deals.csv")
    deals.to_csv(deals_path, index=False)
    print(f"  [OK] {len(deals)} deals saved to {deals_path}")

    # Print some stats so you can verify the data looks realistic
    print(f"\n  Dataset Stats:")
    print(f"  - Win rate: {(deals['outcome'] == 'Won').mean():.1%}")
    print(f"  - Avg deal size: ${deals['deal_amount'].mean():,.0f}")
    print(f"  - Avg days in pipeline: {deals['days_in_pipeline'].mean():.0f}")
    print(f"  - Deals with competitor: {deals['competitor_involved'].mean():.1%}")

    # Outcome distribution
    print(f"\n  Outcome Distribution:")
    for outcome, count in deals["outcome"].value_counts().items():
        print(f"    {outcome}: {count} ({count/len(deals):.1%})")

    # Generate revenue time series
    print("\n[2/3] Generating revenue time series...")
    revenue = generate_revenue_timeseries(deals)
    revenue_path = os.path.join(raw_data_dir, "monthly_revenue.csv")
    revenue.to_csv(revenue_path, index=False)
    print(f"  [OK] {len(revenue)} months of revenue data saved to {revenue_path}")

    # Industry breakdown
    print("\n[3/3] Industry breakdown:")
    industry_stats = deals.groupby("industry").agg(
        avg_deal=("deal_amount", "mean"),
        win_rate=("outcome", lambda x: (x == "Won").mean()),
        count=("deal_id", "count"),
    ).round(2)
    print(industry_stats.to_string())

    print("\n" + "=" * 60)
    print("Data generation complete! Next step: data exploration notebook")
    print("=" * 60)
