"""
SalesIntel AI - Synthetic Sales Documents Generator (for RAG)
==============================================================

WHY DO WE NEED DOCUMENTS?
- The RAG (Retrieval-Augmented Generation) component needs a KNOWLEDGE BASE
  to search through when answering questions
- In a real company, this would be: product sheets, meeting notes, account
  summaries, competitive intel, sales playbooks, etc.
- We generate realistic versions of these so our RAG pipeline has
  something meaningful to retrieve and reason over

WHAT IS RAG? (You'll build this in Week 3)
- RAG = Retrieval-Augmented Generation
- Instead of asking an LLM to answer from its training data (which doesn't
  know about YOUR sales data), you:
    1. RETRIEVE relevant documents from a vector database
    2. AUGMENT the LLM prompt with those documents
    3. GENERATE an answer grounded in your actual data
- The documents we generate here become the knowledge base for step 1

DOCUMENT TYPES WE GENERATE:
1. Account Summaries - overview of each customer account
2. Product Sheets - descriptions of products being sold
3. Meeting Notes - call/meeting summaries with prospects
4. Competitive Intel - info about competitors
5. Sales Playbook - best practices and strategies
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta


def generate_account_summaries(deals_df: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """
    Generate account summary documents from deal data.

    WHY? Account summaries are the #1 thing sales reps look up.
    In a RAG system, when someone asks "Tell me about Apex Technologies",
    this document gets retrieved.

    Each document is a dict with:
    - 'content': the text (what gets embedded into vectors)
    - 'metadata': structured info (used for filtering in retrieval)
    """
    # Group deals by account to create per-account summaries
    accounts = deals_df.groupby("account_name").agg(
        industry=("industry", "first"),
        total_deals=("deal_id", "count"),
        total_revenue=("deal_amount", "sum"),
        avg_deal_size=("deal_amount", "mean"),
        win_rate=("outcome", lambda x: (x == "Won").mean()),
        primary_rep=("sales_rep", "first"),
        latest_stage=("deal_stage", "last"),
    ).reset_index()

    docs = []
    sentiments = ["strong", "growing", "stable", "at risk", "new"]
    engagement_levels = ["highly engaged", "moderately engaged", "low engagement", "re-engaging"]

    for _, acc in accounts.iterrows():
        sentiment = rng.choice(sentiments)
        engagement = rng.choice(engagement_levels)

        content = f"""Account Summary: {acc['account_name']}
Industry: {acc['industry']}
Relationship Status: {sentiment}
Engagement Level: {engagement}
Primary Sales Rep: {acc['primary_rep']}

Overview:
{acc['account_name']} is a {acc['industry'].lower()} company with {acc['total_deals']} deals in our pipeline.
Total revenue potential: ${acc['total_revenue']:,.0f}. Average deal size: ${acc['avg_deal_size']:,.0f}.
Current win rate with this account: {acc['win_rate']:.0%}.

The account is currently {engagement} with our latest opportunity in the {acc['latest_stage']} stage.
Overall relationship sentiment is {sentiment}.

Key Notes:
- {'Decision-making process involves multiple stakeholders' if acc['total_deals'] > 2 else 'Single point of contact for decisions'}
- {'Competitor presence detected in recent deals' if rng.random() > 0.5 else 'No significant competitor activity noted'}
- {'Budget typically approved in Q1 and Q3' if rng.random() > 0.5 else 'Budget cycles align with fiscal year end'}
"""

        docs.append({
            "content": content,
            "metadata": {
                "doc_type": "account_summary",
                "account_name": acc["account_name"],
                "industry": acc["industry"],
                "sales_rep": acc["primary_rep"],
            }
        })

    return docs


def generate_product_sheets() -> list[dict]:
    """
    Generate product/solution descriptions.

    WHY? When the AI assistant is asked "What product fits a manufacturing
    company?", it needs to retrieve product information to give a useful answer.
    """
    products = [
        {
            "name": "SalesIntel Analytics Suite",
            "category": "Analytics",
            "description": "End-to-end sales analytics platform providing real-time pipeline visibility, "
                           "win/loss analysis, and performance benchmarking. Includes customizable dashboards, "
                           "automated reporting, and predictive deal scoring powered by machine learning.",
            "ideal_for": "Mid-market and enterprise companies with 50+ sales reps",
            "price_range": "$30,000 - $150,000 annually",
            "key_features": [
                "Real-time pipeline dashboards",
                "AI-powered deal scoring",
                "Win/loss pattern analysis",
                "Rep performance benchmarking",
                "Custom report builder",
            ],
        },
        {
            "name": "SmartForecast Pro",
            "category": "Forecasting",
            "description": "AI-driven revenue forecasting solution that combines historical deal data, "
                           "pipeline signals, and market trends to deliver accurate quarterly and annual "
                           "revenue predictions. Reduces forecast error by up to 35%.",
            "ideal_for": "Sales leaders and finance teams needing accurate revenue projections",
            "price_range": "$20,000 - $80,000 annually",
            "key_features": [
                "ML-based revenue forecasting",
                "Scenario modeling (best/worst/likely)",
                "Pipeline health indicators",
                "Forecast accuracy tracking",
                "Integration with major CRMs",
            ],
        },
        {
            "name": "DealFlow Automation",
            "category": "Automation",
            "description": "Intelligent workflow automation for sales processes. Automates lead routing, "
                           "follow-up scheduling, data entry, and approval workflows. Helps sales reps "
                           "spend more time selling and less time on admin tasks.",
            "ideal_for": "Sales teams looking to reduce manual work and improve response times",
            "price_range": "$15,000 - $60,000 annually",
            "key_features": [
                "Automated lead scoring and routing",
                "Smart follow-up reminders",
                "Email template automation",
                "Approval workflow engine",
                "CRM data enrichment",
            ],
        },
        {
            "name": "CompetitiveEdge Intel",
            "category": "Competitive Intelligence",
            "description": "Real-time competitive intelligence platform that tracks competitor pricing, "
                           "product launches, and market positioning. Provides battle cards and talking "
                           "points for sales reps to handle competitive objections.",
            "ideal_for": "Sales teams in competitive markets needing real-time competitive insights",
            "price_range": "$10,000 - $40,000 annually",
            "key_features": [
                "Competitor tracking dashboard",
                "AI-generated battle cards",
                "Win/loss competitive analysis",
                "Pricing intelligence",
                "Objection handling playbooks",
            ],
        },
    ]

    docs = []
    for product in products:
        features_text = "\n".join(f"  - {f}" for f in product["key_features"])
        content = f"""Product Sheet: {product['name']}
Category: {product['category']}
Price Range: {product['price_range']}

Description:
{product['description']}

Ideal For:
{product['ideal_for']}

Key Features:
{features_text}
"""
        docs.append({
            "content": content,
            "metadata": {
                "doc_type": "product_sheet",
                "product_name": product["name"],
                "category": product["category"],
            }
        })

    return docs


def generate_meeting_notes(deals_df: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """
    Generate synthetic meeting/call notes.

    WHY? Meeting notes are rich, unstructured text - exactly the kind of
    content that RAG excels at searching through. When someone asks
    "What concerns did Apex Technologies raise?", the system retrieves
    relevant meeting notes.
    """
    # Pick a subset of deals to generate meeting notes for
    sample_deals = deals_df.sample(n=min(200, len(deals_df)), random_state=42)

    topics = [
        "product demo and feature walkthrough",
        "pricing discussion and budget alignment",
        "technical requirements and integration planning",
        "stakeholder alignment and next steps",
        "contract review and negotiation",
        "quarterly business review",
        "competitive comparison discussion",
        "implementation timeline planning",
    ]

    concerns = [
        "integration with existing ERP system",
        "data migration complexity",
        "user adoption and training requirements",
        "total cost of ownership over 3 years",
        "security and compliance requirements",
        "scalability for future growth",
        "support and SLA guarantees",
        "customization capabilities",
    ]

    next_steps_options = [
        "Schedule technical deep-dive with IT team",
        "Send revised pricing proposal",
        "Arrange reference call with similar customer",
        "Provide ROI analysis document",
        "Set up proof-of-concept environment",
        "Schedule follow-up with VP of Sales",
        "Send contract for legal review",
        "Prepare implementation timeline",
    ]

    docs = []
    for _, deal in sample_deals.iterrows():
        topic = rng.choice(topics)
        concern = rng.choice(concerns)
        next_step = rng.choice(next_steps_options)
        meeting_date = deal["created_date"] + timedelta(days=int(rng.integers(1, 30)))
        sentiment = rng.choice(["positive", "neutral", "cautious", "enthusiastic"])

        content = f"""Meeting Notes - {deal['account_name']}
Date: {meeting_date.strftime('%Y-%m-%d')}
Sales Rep: {deal['sales_rep']}
Deal: {deal['deal_id']} (${deal['deal_amount']:,.0f})
Stage: {deal['deal_stage']}

Topic: {topic}

Summary:
Met with {deal['account_name']} ({deal['industry']}) to discuss {topic}.
Overall tone of the meeting was {sentiment}. The deal is currently valued at
${deal['deal_amount']:,.0f} and is in the {deal['deal_stage']} stage.

Key Discussion Points:
- Reviewed current challenges and how our solution addresses their needs
- Discussed {topic} in detail
- Client raised concerns about {concern}
- {'Competitor was mentioned as an alternative being evaluated' if deal['competitor_involved'] else 'No competitor mentioned during the discussion'}

Client Concerns:
- Primary concern: {concern}
- {'Budget has been pre-approved' if rng.random() > 0.5 else 'Budget approval still pending'}

Next Steps:
- {next_step}
- Follow-up scheduled for {(meeting_date + timedelta(days=int(rng.integers(3, 14)))).strftime('%Y-%m-%d')}
"""

        docs.append({
            "content": content,
            "metadata": {
                "doc_type": "meeting_notes",
                "account_name": deal["account_name"],
                "deal_id": deal["deal_id"],
                "sales_rep": deal["sales_rep"],
                "meeting_date": meeting_date.strftime("%Y-%m-%d"),
            }
        })

    return docs


def generate_sales_playbook() -> list[dict]:
    """
    Generate sales playbook / best-practices documents.

    WHY? This gives the RAG assistant the ability to answer strategic
    questions like "How should I handle a competitor objection?" or
    "What's the best approach for enterprise deals?"
    """
    playbook_sections = [
        {
            "title": "Handling Competitive Objections",
            "content": """Sales Playbook: Handling Competitive Objections

When a prospect mentions a competitor, follow the ACE framework:
- Acknowledge: "I understand you're evaluating other options - that's smart."
- Compare: Focus on differentiation, not bashing. Highlight unique capabilities.
- Evidence: Share relevant case studies and ROI metrics from similar customers.

Common Competitor Objections:
1. "Competitor X is cheaper" → Reframe to total cost of ownership and time-to-value
2. "We already use Competitor Y" → Focus on switching costs vs. long-term gains
3. "Competitor Z has feature ABC" → Demonstrate how our approach solves the underlying need differently

Key Stats to Reference:
- Our customers see 25% faster implementation than industry average
- 92% customer retention rate (vs. industry average of 78%)
- Average ROI of 340% over 3 years based on customer surveys
"""
        },
        {
            "title": "Enterprise Deal Strategy",
            "content": """Sales Playbook: Enterprise Deal Strategy

Enterprise deals ($100K+) require a multi-threaded approach:

1. Stakeholder Mapping:
   - Identify the Economic Buyer (controls budget)
   - Find the Champion (internal advocate)
   - Map the Influencers (technical evaluators)
   - Note any Blockers (resistant to change)

2. Deal Progression Framework:
   - Discovery: Understand pain points, current solutions, and desired outcomes
   - Qualification: Confirm BANT (Budget, Authority, Need, Timeline)
   - Solution Design: Tailor proposal to specific use cases
   - Proof of Value: POC or pilot with measurable success criteria
   - Negotiation: Focus on value, not just price
   - Close: Ensure all stakeholders are aligned before final contract

3. Red Flags to Watch:
   - No access to Economic Buyer after 2nd meeting
   - Prospect won't commit to POC timeline
   - Single-threaded (only one contact)
   - No defined evaluation criteria
   - "We'll get back to you" with no specific date
"""
        },
        {
            "title": "Quarterly Forecasting Best Practices",
            "content": """Sales Playbook: Forecasting Best Practices

Accurate forecasting is critical for business planning. Follow these guidelines:

Deal Categories:
- Commit: 90%+ probability, verbal agreement, contract in review
- Best Case: 60-89% probability, strong engagement, proposal accepted
- Pipeline: 30-59% probability, active opportunity, needs identified
- Upside: <30% probability, early stage, not yet qualified

Weekly Forecast Hygiene:
1. Review ALL deals in Commit category - can you defend each one?
2. Move stale deals (no activity in 30+ days) down or out
3. Update close dates to reflect reality, not optimism
4. Document next steps for every deal in Best Case or above

Common Forecasting Mistakes:
- Including deals with no recent activity
- Optimistic close dates ("it'll close this quarter" without evidence)
- Not accounting for procurement/legal timelines
- Ignoring historical conversion rates by stage
"""
        },
    ]

    docs = []
    for section in playbook_sections:
        docs.append({
            "content": section["content"],
            "metadata": {
                "doc_type": "sales_playbook",
                "section": section["title"],
            }
        })

    return docs


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SalesIntel AI - Generating Sales Documents for RAG")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    raw_data_dir = os.path.join(project_root, "data", "raw")
    docs_dir = os.path.join(project_root, "data", "docs")
    os.makedirs(docs_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    # Load deals data (must run generate_deals.py first)
    deals_path = os.path.join(raw_data_dir, "deals.csv")
    if not os.path.exists(deals_path):
        print("ERROR: deals.csv not found. Run generate_deals.py first!")
        exit(1)

    deals_df = pd.read_csv(deals_path, parse_dates=["created_date", "expected_close_date"])
    print(f"  Loaded {len(deals_df)} deals from {deals_path}")

    # Generate all document types
    all_docs = []

    print("\n[1/4] Generating account summaries...")
    account_docs = generate_account_summaries(deals_df, rng)
    all_docs.extend(account_docs)
    print(f"  [OK] {len(account_docs)} account summaries")

    print("[2/4] Generating product sheets...")
    product_docs = generate_product_sheets()
    all_docs.extend(product_docs)
    print(f"  [OK] {len(product_docs)} product sheets")

    print("[3/4] Generating meeting notes...")
    meeting_docs = generate_meeting_notes(deals_df, rng)
    all_docs.extend(meeting_docs)
    print(f"  [OK] {len(meeting_docs)} meeting notes")

    print("[4/4] Generating sales playbook...")
    playbook_docs = generate_sales_playbook()
    all_docs.extend(playbook_docs)
    print(f"  [OK] {len(playbook_docs)} playbook sections")

    # Save all documents as JSON
    # WHY JSON? It preserves the metadata structure.
    # When we build the RAG pipeline, we'll load these, embed the 'content'
    # field into vectors, and store 'metadata' for filtering.
    docs_path = os.path.join(docs_dir, "sales_documents.json")
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, default=str)
    print(f"\n  [OK] Total: {len(all_docs)} documents saved to {docs_path}")

    # Also save individual text files for easy browsing
    print("\n  Saving individual text files for easy reading...")
    for doc_type in ["account_summary", "product_sheet", "meeting_notes", "sales_playbook"]:
        type_dir = os.path.join(docs_dir, doc_type)
        os.makedirs(type_dir, exist_ok=True)
        type_docs = [d for d in all_docs if d["metadata"]["doc_type"] == doc_type]
        for i, doc in enumerate(type_docs):
            filename = f"{doc_type}_{i+1:03d}.txt"
            with open(os.path.join(type_dir, filename), "w", encoding="utf-8") as f:
                f.write(doc["content"])

    print("\n" + "=" * 60)
    print("Document generation complete!")
    print(f"Total documents: {len(all_docs)}")
    print("=" * 60)
