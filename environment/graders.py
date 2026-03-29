"""
Agent graders for all three tasks.
Each grader scores a completed episode 0.0–1.0 with structured partial credit.
All graders are deterministic and reproducible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Priority adjacency for partial credit (task 2 & 3)
# ---------------------------------------------------------------------------

PRIORITY_ORDER = ["urgent", "high", "medium", "low"]

DEPARTMENT_MAP = {
    "billing": "billing_team",
    "technical": "tech_support",
    "account": "account_management",
    "general": "general_support",
    "feature_request": "general_support",
}


def _priority_score(predicted: str, correct: str) -> float:
    """
    Full credit for exact match; partial for adjacent levels.
    urgent↔high=0.5, high↔medium=0.5, medium↔low=0.5, else=0.0
    """
    if predicted == correct:
        return 1.0
    try:
        pi = PRIORITY_ORDER.index(predicted)
        ci = PRIORITY_ORDER.index(correct)
        if abs(pi - ci) == 1:
            return 0.5
    except ValueError:
        pass
    return 0.0


def _response_quality_score(
    response: str,
    keywords: List[str],
    min_words: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Heuristic response quality scorer (no LLM needed — deterministic).

    Dimensions:
    - Length adequacy  (25%) — meets minimum word count
    - Keyword coverage (40%) — addresses key topic areas
    - Professionalism  (20%) — has greeting + closing
    - Empathy markers  (15%) — uses empathetic language
    """
    if not response or not response.strip():
        return 0.0, {"length": 0.0, "keywords": 0.0, "professionalism": 0.0, "empathy": 0.0}

    response_lower = response.lower()
    words = response.split()

    # Length score
    length_score = min(1.0, len(words) / min_words)

    # Keyword coverage
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    keyword_score = matched / max(len(keywords), 1)

    # Professionalism — has a greeting and a closing
    greetings = ["hello", "hi ", "dear", "thank you for", "thanks for reaching"]
    closings = ["regards", "sincerely", "best", "let us know", "please don't hesitate",
                "feel free", "happy to help", "here to help"]
    has_greeting = any(g in response_lower for g in greetings)
    has_closing = any(c in response_lower for c in closings)
    professionalism_score = (0.5 * int(has_greeting)) + (0.5 * int(has_closing))

    # Empathy
    empathy_phrases = [
        "understand", "apolog", "sorry", "frustrat", "appreciate",
        "concern", "important", "priority", "right away",
    ]
    empathy_matches = sum(1 for p in empathy_phrases if p in response_lower)
    empathy_score = min(1.0, empathy_matches / 3)

    breakdown = {
        "length": length_score,
        "keywords": keyword_score,
        "professionalism": professionalism_score,
        "empathy": empathy_score,
    }

    total = (
        0.25 * length_score
        + 0.40 * keyword_score
        + 0.20 * professionalism_score
        + 0.15 * empathy_score
    )
    return round(total, 4), breakdown


# ---------------------------------------------------------------------------
# Task 1 Grader — Classification
# ---------------------------------------------------------------------------

def grade_task1(action_log: List[Dict[str, Any]], tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score = fraction of tickets classified into the correct category.
    No partial credit for wrong categories.
    Returns detailed per-ticket breakdown.
    """
    classify_actions = [a for a in action_log if a.get("action_type") == "classify"]
    if not classify_actions:
        return {"score": 0.0, "reason": "No classify actions taken", "details": []}

    correct = 0
    details = []
    for i, ticket in enumerate(tickets):
        predicted = classify_actions[i]["category"] if i < len(classify_actions) else None
        expected = ticket["correct_category"]
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        details.append({
            "ticket_id": ticket["id"],
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
        })

    score = round(correct / len(tickets), 4)
    return {
        "score": score,
        "reason": f"Correctly classified {correct}/{len(tickets)} tickets",
        "details": details,
    }


# ---------------------------------------------------------------------------
# Task 2 Grader — Priority & Routing
# ---------------------------------------------------------------------------

def grade_task2(action_log: List[Dict[str, Any]], tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score = 0.5 * priority_accuracy + 0.5 * routing_accuracy
    Priority uses partial-credit scoring; routing is exact match.
    """
    prioritize_actions = [a for a in action_log if a.get("action_type") == "prioritize"]
    route_actions = [a for a in action_log if a.get("action_type") == "route"]

    priority_scores = []
    routing_scores = []
    details = []

    for i, ticket in enumerate(tickets):
        pred_priority = prioritize_actions[i]["priority"] if i < len(prioritize_actions) else None
        pred_dept = route_actions[i]["department"] if i < len(route_actions) else None

        exp_priority = ticket["correct_priority"]
        exp_dept = ticket["correct_department"]

        p_score = _priority_score(pred_priority or "", exp_priority)
        r_score = 1.0 if pred_dept == exp_dept else 0.0

        priority_scores.append(p_score)
        routing_scores.append(r_score)

        details.append({
            "ticket_id": ticket["id"],
            "priority": {"predicted": pred_priority, "expected": exp_priority, "score": p_score},
            "routing": {"predicted": pred_dept, "expected": exp_dept, "score": r_score},
        })

    n = len(tickets)
    avg_priority = sum(priority_scores) / n if n else 0.0
    avg_routing = sum(routing_scores) / n if n else 0.0
    score = round(0.5 * avg_priority + 0.5 * avg_routing, 4)

    return {
        "score": score,
        "reason": (
            f"Priority accuracy: {avg_priority:.2f}, Routing accuracy: {avg_routing:.2f}"
        ),
        "details": details,
        "component_scores": {
            "priority_accuracy": round(avg_priority, 4),
            "routing_accuracy": round(avg_routing, 4),
        },
    }


# ---------------------------------------------------------------------------
# Task 3 Grader — Full Resolution
# ---------------------------------------------------------------------------

def grade_task3(action_log: List[Dict[str, Any]], tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Per ticket: classify (20%) + prioritize (20%) + route (20%) + respond (40%)
    Final score = mean across all tickets.
    """
    ticket_scores = []
    details = []

    classify_actions = [a for a in action_log if a.get("action_type") == "classify"]
    prioritize_actions = [a for a in action_log if a.get("action_type") == "prioritize"]
    route_actions = [a for a in action_log if a.get("action_type") == "route"]
    respond_actions = [a for a in action_log if a.get("action_type") == "respond"]

    for i, ticket in enumerate(tickets):
        pred_cat = classify_actions[i]["category"] if i < len(classify_actions) else None
        pred_pri = prioritize_actions[i]["priority"] if i < len(prioritize_actions) else None
        pred_dept = route_actions[i]["department"] if i < len(route_actions) else None
        pred_resp = respond_actions[i].get("response_text", "") if i < len(respond_actions) else ""

        cat_score = 1.0 if pred_cat == ticket["correct_category"] else 0.0
        pri_score = _priority_score(pred_pri or "", ticket["correct_priority"])
        dept_score = 1.0 if pred_dept == ticket["correct_department"] else 0.0
        resp_score, resp_breakdown = _response_quality_score(
            pred_resp,
            ticket.get("response_keywords", []),
            ticket.get("response_min_words", 60),
        )

        ticket_score = (
            0.20 * cat_score
            + 0.20 * pri_score
            + 0.20 * dept_score
            + 0.40 * resp_score
        )
        ticket_scores.append(ticket_score)

        details.append({
            "ticket_id": ticket["id"],
            "classification": {"predicted": pred_cat, "expected": ticket["correct_category"], "score": cat_score},
            "priority": {"predicted": pred_pri, "expected": ticket["correct_priority"], "score": pri_score},
            "routing": {"predicted": pred_dept, "expected": ticket["correct_department"], "score": dept_score},
            "response": {"score": resp_score, "breakdown": resp_breakdown, "word_count": len((pred_resp or "").split())},
            "ticket_total": round(ticket_score, 4),
        })

    score = round(sum(ticket_scores) / len(tickets), 4) if tickets else 0.0

    return {
        "score": score,
        "reason": f"Average full-resolution score across {len(tickets)} tickets",
        "details": details,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}


def run_grader(
    task_id: str,
    action_log: List[Dict[str, Any]],
    tickets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the appropriate grader for the given task and return structured results."""
    grader = GRADERS.get(task_id)
    if grader is None:
        return {"score": 0.0, "reason": f"Unknown task_id: {task_id}", "details": []}
    result = grader(action_log, tickets)
    result["task_id"] = task_id
    return result
