"""
Task definitions for the Customer Support Triage environment.
Each task has metadata, instructions, valid actions, and step budgets.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .data import TASK1_TICKETS, TASK2_TICKETS, TASK3_TICKETS


TASK_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # TASK 1 — Easy: Pure classification
    # ------------------------------------------------------------------
    "task1": {
        "id": "task1",
        "name": "Ticket Classification",
        "difficulty": "easy",
        "description": (
            "You are a Tier-1 support agent. For each incoming ticket, "
            "classify it into exactly one category:\n"
            "  • billing         — payment, invoice, refund, charge issues\n"
            "  • technical       — bugs, API errors, performance problems\n"
            "  • account         — login, access, account settings, data loss\n"
            "  • general         — how-to questions, documentation requests\n"
            "  • feature_request — requests for new product capabilities\n\n"
            "Use action_type='classify' with the appropriate category for each ticket."
        ),
        "valid_action_types": ["classify", "skip"],
        "tickets": TASK1_TICKETS,
        "max_steps": 14,  # 2x tickets for retries
        "scoring": {
            "method": "exact_match",
            "weight_per_ticket": 1.0 / len(TASK1_TICKETS),
        },
        "reward_config": {
            "correct_classify": 1.0 / len(TASK1_TICKETS),
            "wrong_classify": -0.05,
            "skip_penalty": -0.02,
        },
    },

    # ------------------------------------------------------------------
    # TASK 2 — Medium: Priority + routing (category pre-filled)
    # ------------------------------------------------------------------
    "task2": {
        "id": "task2",
        "name": "Priority & Routing",
        "difficulty": "medium",
        "description": (
            "You are a Tier-2 support routing agent. Each ticket's category "
            "has already been set. Your job is to:\n"
            "  1. Set the correct PRIORITY: urgent | high | medium | low\n"
            "     • urgent  — production down, security incident, SLA breach imminent\n"
            "     • high    — significant customer impact, revenue at risk\n"
            "     • medium  — moderate impact, workaround exists\n"
            "     • low     — cosmetic, informational, nice-to-have\n"
            "  2. ROUTE to the correct department:\n"
            "     • billing_team         — for billing/payment issues\n"
            "     • tech_support         — for technical/API/bug issues\n"
            "     • account_management   — for account, access, upgrade issues\n"
            "     • general_support      — for general and feature requests\n\n"
            "Use action_type='prioritize' then action_type='route' for each ticket."
        ),
        "valid_action_types": ["prioritize", "route", "skip"],
        "tickets": TASK2_TICKETS,
        "max_steps": 20,  # 4 actions per ticket + buffer
        "scoring": {
            "method": "weighted_priority_routing",
            "priority_weight": 0.5,
            "routing_weight": 0.5,
        },
        "reward_config": {
            "correct_priority": 0.5 / len(TASK2_TICKETS),
            "adjacent_priority": 0.25 / len(TASK2_TICKETS),
            "wrong_priority": -0.03,
            "correct_routing": 0.5 / len(TASK2_TICKETS),
            "wrong_routing": -0.03,
        },
    },

    # ------------------------------------------------------------------
    # TASK 3 — Hard: Full resolution
    # ------------------------------------------------------------------
    "task3": {
        "id": "task3",
        "name": "Full Ticket Resolution",
        "difficulty": "hard",
        "description": (
            "You are a senior support agent handling escalated tickets. "
            "For each ticket you must:\n"
            "  1. CLASSIFY into the primary category\n"
            "  2. PRIORITIZE with appropriate urgency\n"
            "  3. ROUTE to the correct department\n"
            "  4. RESPOND with a complete, professional customer reply\n\n"
            "Tickets are complex, multi-issue, and emotionally charged. "
            "Your response will be scored on:\n"
            "  • Addressing the core issue(s)\n"
            "  • Empathy and professionalism\n"
            "  • Completeness and actionability\n"
            "  • Appropriate length (minimum 60–100 words depending on ticket)\n\n"
            "Use action_type='classify', 'prioritize', 'route', then 'respond' for each ticket."
        ),
        "valid_action_types": ["classify", "prioritize", "route", "respond", "skip"],
        "tickets": TASK3_TICKETS,
        "max_steps": 20,  # 4+ actions per ticket + buffer
        "scoring": {
            "method": "full_resolution",
            "weights": {
                "classify": 0.20,
                "prioritize": 0.20,
                "route": 0.20,
                "respond": 0.40,
            },
        },
        "reward_config": {
            "correct_classify": 0.20 / len(TASK3_TICKETS),
            "correct_priority": 0.20 / len(TASK3_TICKETS),
            "correct_routing": 0.20 / len(TASK3_TICKETS),
            "response_quality_max": 0.40 / len(TASK3_TICKETS),
            "wrong_any": -0.02,
        },
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_DEFINITIONS.keys())}")
    return TASK_DEFINITIONS[task_id]


def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "num_tickets": len(t["tickets"]),
            "max_steps": t["max_steps"],
        }
        for t in TASK_DEFINITIONS.values()
    ]
