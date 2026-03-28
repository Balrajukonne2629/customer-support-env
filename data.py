"""
Curated dataset of realistic customer support tickets.
Each ticket has ground-truth labels used by the graders.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# TASK 1 — Ticket Classification (Easy)
# Agent must classify each ticket into the correct category.
# ---------------------------------------------------------------------------

TASK1_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "T1-001",
        "subject": "Charged twice this billing cycle",
        "body": (
            "Hello, I just reviewed my bank statement and see two charges of $49.99 "
            "from your company dated March 1st and March 3rd. I should only be billed once "
            "per month. Please investigate and refund the duplicate charge. "
            "My account email is john.doe@example.com."
        ),
        "customer_tier": "premium",
        "timestamp": "2024-03-05T08:15:00Z",
        "sla_hours": 4,
        "correct_category": "billing",
        "correct_priority": "high",
        "correct_department": "billing_team",
    },
    {
        "id": "T1-002",
        "subject": "API returning 500 errors intermittently",
        "body": (
            "Our production system has been hitting 500 Internal Server Error responses "
            "from your /v2/data endpoint since 6 AM UTC. This is happening for roughly "
            "30% of requests. We have an enterprise SLA and this is causing customer-facing "
            "downtime. Request IDs: req_8fh3k, req_9xjal, req_0plmn."
        ),
        "customer_tier": "enterprise",
        "timestamp": "2024-03-05T09:02:00Z",
        "sla_hours": 1,
        "correct_category": "technical",
        "correct_priority": "urgent",
        "correct_department": "tech_support",
    },
    {
        "id": "T1-003",
        "subject": "Cannot log in — says account does not exist",
        "body": (
            "I'm trying to log in with my email sarah.j@company.org but the system says "
            "'account not found'. I've been a customer for 2 years. Did my account get "
            "deleted? I haven't changed anything on my end. I need access urgently as I "
            "have a presentation tomorrow."
        ),
        "customer_tier": "basic",
        "timestamp": "2024-03-05T10:30:00Z",
        "sla_hours": 8,
        "correct_category": "account",
        "correct_priority": "high",
        "correct_department": "account_management",
    },
    {
        "id": "T1-004",
        "subject": "How do I export data to CSV?",
        "body": (
            "Hi, I'd like to export all my project data as a CSV file to share with "
            "my team. I looked in the Settings menu but couldn't find an export option. "
            "Is this feature available? If so, where can I find it? Thanks!"
        ),
        "customer_tier": "free",
        "timestamp": "2024-03-05T11:45:00Z",
        "sla_hours": 48,
        "correct_category": "general",
        "correct_priority": "low",
        "correct_department": "general_support",
    },
    {
        "id": "T1-005",
        "subject": "Would love a dark mode option",
        "body": (
            "Hey team! Love the product. One thing that would make it even better: "
            "a dark mode for the dashboard. I work late nights and the bright white "
            "interface is tough on the eyes. Has this been considered? Would be amazing "
            "to have this in a future release."
        ),
        "customer_tier": "basic",
        "timestamp": "2024-03-05T13:00:00Z",
        "sla_hours": 72,
        "correct_category": "feature_request",
        "correct_priority": "low",
        "correct_department": "general_support",
    },
    {
        "id": "T1-006",
        "subject": "Invoice shows wrong company name",
        "body": (
            "The invoice I received for February shows 'Acme Corp' as the billing name "
            "but our legal entity name is 'Acme Corporation Ltd'. This is causing issues "
            "with our accounting department. Please update and resend a corrected invoice."
        ),
        "customer_tier": "enterprise",
        "timestamp": "2024-03-05T14:10:00Z",
        "sla_hours": 8,
        "correct_category": "billing",
        "correct_priority": "medium",
        "correct_department": "billing_team",
    },
    {
        "id": "T1-007",
        "subject": "Webhook payloads stopped arriving",
        "body": (
            "Our webhook endpoint (https://hooks.myapp.com/incoming) stopped receiving "
            "events about 3 hours ago. I've verified our endpoint is healthy and returning "
            "200s. Last successful event was at 11:23 UTC. We're missing critical order "
            "fulfillment events. Please check on your side."
        ),
        "customer_tier": "premium",
        "timestamp": "2024-03-05T14:30:00Z",
        "sla_hours": 2,
        "correct_category": "technical",
        "correct_priority": "urgent",
        "correct_department": "tech_support",
    },
]


# ---------------------------------------------------------------------------
# TASK 2 — Priority & Routing (Medium)
# Category is pre-filled; agent must set priority + department correctly.
# ---------------------------------------------------------------------------

TASK2_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "T2-001",
        "subject": "Complete platform outage — nothing loads",
        "body": (
            "CRITICAL: None of our 500 users can access the platform. Dashboard shows "
            "blank screen after login. This has been going on for 20 minutes. We are "
            "an enterprise customer with 99.9% uptime SLA. This breach will incur penalties."
        ),
        "customer_tier": "enterprise",
        "timestamp": "2024-03-06T07:00:00Z",
        "sla_hours": 1,
        "category": "technical",
        "correct_category": "technical",
        "correct_priority": "urgent",
        "correct_department": "tech_support",
    },
    {
        "id": "T2-002",
        "subject": "Need annual invoice for tax filing",
        "body": (
            "Hi, I need an annual invoice for all my 2023 payments for my tax filing. "
            "The deadline is next week. Could you generate a consolidated invoice for "
            "the full year? My account is under mike.chen@startup.io."
        ),
        "customer_tier": "basic",
        "timestamp": "2024-03-06T09:00:00Z",
        "sla_hours": 48,
        "category": "billing",
        "correct_category": "billing",
        "correct_priority": "medium",
        "correct_department": "billing_team",
    },
    {
        "id": "T2-003",
        "subject": "Want to upgrade to Enterprise plan",
        "body": (
            "Our company is growing fast and we'd like to discuss upgrading to the "
            "Enterprise plan. We currently have 45 users and need SSO, audit logs, and "
            "dedicated support. Who should I speak with about pricing and migration?"
        ),
        "customer_tier": "premium",
        "timestamp": "2024-03-06T10:30:00Z",
        "sla_hours": 24,
        "category": "account",
        "correct_category": "account",
        "correct_priority": "high",
        "correct_department": "account_management",
    },
    {
        "id": "T2-004",
        "subject": "Slow performance on large reports",
        "body": (
            "When I generate reports with more than 10,000 rows, the page freezes for "
            "about 30-45 seconds before showing results. This started a few days ago. "
            "Smaller reports work fine. Running Chrome on Windows 11."
        ),
        "customer_tier": "free",
        "timestamp": "2024-03-06T11:00:00Z",
        "sla_hours": 72,
        "category": "technical",
        "correct_category": "technical",
        "correct_priority": "medium",
        "correct_department": "tech_support",
    },
    {
        "id": "T2-005",
        "subject": "Accidentally deleted a project — can it be restored?",
        "body": (
            "I accidentally deleted our main project about an hour ago. It contained "
            "6 months of work. Is there a way to restore it from a backup? This is very "
            "urgent — our team is blocked and we have a client demo tomorrow morning."
        ),
        "customer_tier": "premium",
        "timestamp": "2024-03-06T13:00:00Z",
        "sla_hours": 2,
        "category": "account",
        "correct_category": "account",
        "correct_priority": "urgent",
        "correct_department": "account_management",
    },
]


# ---------------------------------------------------------------------------
# TASK 3 — Full Resolution (Hard)
# Agent must classify, prioritize, route, AND write a quality response.
# These tickets are complex, multi-issue, and require nuanced handling.
# ---------------------------------------------------------------------------

TASK3_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "T3-001",
        "subject": "Multiple issues: billing error + API failures + angry",
        "body": (
            "I am EXTREMELY frustrated. For the third month in a row, I have been "
            "overcharged — this time by $150. On top of that, your API has been "
            "returning 429 rate limit errors even though I'm well within my quota "
            "of 10,000 requests/day (I've only made 3,000 today). I have been a "
            "premium customer for 4 years and this kind of service is unacceptable. "
            "If this isn't resolved by EOD I'm cancelling and filing a chargeback."
        ),
        "customer_tier": "premium",
        "timestamp": "2024-03-07T08:00:00Z",
        "sla_hours": 2,
        "previous_messages": [
            "Agent (Feb): Apologies for the billing issue, we've applied a credit.",
            "Agent (Jan): We've investigated and refunded the overcharge.",
        ],
        "correct_category": "billing",  # primary issue
        "correct_priority": "urgent",
        "correct_department": "billing_team",
        "response_keywords": [
            "apolog", "billing", "refund", "rate limit", "investigate",
            "priorit", "escalat", "4 year", "loyal", "credit",
        ],
        "response_min_words": 80,
    },
    {
        "id": "T3-002",
        "subject": "Data breach concern — saw unexpected login from Russia",
        "body": (
            "I received an email saying someone logged into my account from Moscow, "
            "Russia at 3:14 AM. I did not do this. I've already changed my password "
            "and enabled 2FA just now. However I'm worried my data may have been "
            "accessed or exported. We store sensitive client data in the platform. "
            "What data did this unknown user access? Are there audit logs I can review? "
            "Do I need to notify my clients under GDPR?"
        ),
        "customer_tier": "enterprise",
        "timestamp": "2024-03-07T09:30:00Z",
        "sla_hours": 1,
        "previous_messages": [],
        "correct_category": "account",
        "correct_priority": "urgent",
        "correct_department": "account_management",
        "response_keywords": [
            "secur", "investigat", "audit log", "escalat", "2fa",
            "gdpr", "data access", "urgently", "team", "account",
        ],
        "response_min_words": 100,
    },
    {
        "id": "T3-003",
        "subject": "Requesting discount for annual plan renewal",
        "body": (
            "Our annual subscription renews on April 1st for $2,400. Given that we've "
            "been customers for 3 years and refer roughly 5 new customers per year, "
            "I'd like to discuss whether a loyalty discount is possible. We're also "
            "evaluating competitors who are offering 20% off for annual plans. "
            "We love the product but budget is tight this year."
        ),
        "customer_tier": "premium",
        "timestamp": "2024-03-07T11:00:00Z",
        "sla_hours": 48,
        "previous_messages": [],
        "correct_category": "billing",
        "correct_priority": "high",
        "correct_department": "account_management",
        "response_keywords": [
            "thank", "loyal", "3 year", "discount", "renew", "team",
            "discuss", "value", "appreciat", "plan",
        ],
        "response_min_words": 60,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_TICKET_MAP: Dict[str, List[Dict[str, Any]]] = {
    "task1": TASK1_TICKETS,
    "task2": TASK2_TICKETS,
    "task3": TASK3_TICKETS,
}
