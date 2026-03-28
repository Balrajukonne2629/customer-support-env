"""
inference.py — Baseline inference script for Customer Support Triage OpenEnv.

Uses the OpenAI client to run an LLM agent against all three tasks and
produces reproducible baseline scores.

Environment variables required:
  API_BASE_URL   — LLM API endpoint (OpenAI-compatible)
  MODEL_NAME     — Model identifier
  HF_TOKEN       — Hugging Face / API key (used as API key)

Usage:
  python inference.py
  python inference.py --task task1        # single task
  python inference.py --host http://localhost:7860  # custom env URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
ENV_HOST: str = os.environ.get("ENV_HOST", "http://localhost:7860")

TEMPERATURE: float = 0.0   # deterministic for reproducibility
MAX_TOKENS: int = 512
MAX_STEPS: int = 30         # safety cap

SYSTEM_PROMPT = """You are an expert customer support agent AI. You will receive customer support tickets and must process them using structured actions.

For each ticket, respond with a JSON object using this exact format:
{
  "action_type": "<classify|prioritize|route|respond>",
  "category": "<billing|technical|account|general|feature_request>",   // only for classify
  "priority": "<urgent|high|medium|low>",                               // only for prioritize
  "department": "<billing_team|tech_support|account_management|general_support>",  // only for route
  "response_text": "<your full customer reply>"                          // only for respond
}

Rules:
- For task1: use action_type="classify" for each ticket
- For task2: first use action_type="prioritize", then "route" for each ticket  
- For task3: use "classify", then "prioritize", then "route", then "respond" for each ticket
- Always output ONLY valid JSON — no explanation, no markdown, no extra text
- For responses: be professional, empathetic, address the specific issue, minimum 80 words for complex tickets

Category guide:
- billing: payments, invoices, charges, refunds
- technical: bugs, API errors, performance, outages  
- account: login, access, data loss, account settings
- general: how-to questions, documentation
- feature_request: requests for new features

Priority guide:
- urgent: production down, security incident, SLA breach imminent
- high: significant customer impact, revenue at risk
- medium: moderate impact, workaround available
- low: minor, cosmetic, informational

Department routing:
- billing_team → billing/payment issues
- tech_support → technical/API/bug issues
- account_management → account, access, upgrade issues
- general_support → general questions and feature requests
"""

# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------

def env_reset(task_id: str, host: str) -> Dict[str, Any]:
    resp = requests.post(f"{host}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any], host: str) -> Dict[str, Any]:
    resp = requests.post(f"{host}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_state(host: str) -> Dict[str, Any]:
    resp = requests.get(f"{host}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    """Call the LLM and return the raw text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM output into an action dict. Returns None on failure."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def build_user_message(observation: Dict[str, Any]) -> str:
    """Format the current observation into a user message for the LLM."""
    ticket = observation.get("ticket")
    if ticket is None:
        return "No more tickets. Episode is done."

    ctx = observation.get("context", {})
    task_id = observation.get("task_id", "")
    valid_actions = observation.get("valid_action_types", [])

    parts = [
        f"TASK: {observation.get('task_id', '').upper()}",
        f"Ticket {ctx.get('ticket_number', '?')}/{ctx.get('total_tickets', '?')}",
        "",
        f"Subject: {ticket['subject']}",
        f"Customer tier: {ticket['customer_tier']}",
        f"SLA hours remaining: {ticket.get('sla_hours', 'N/A')}",
        f"Message:\n{ticket['body']}",
    ]

    if ticket.get("previous_messages"):
        parts.append("\nPrevious conversation:")
        for msg in ticket["previous_messages"]:
            parts.append(f"  {msg}")

    if ctx.get("pre_filled_category"):
        parts.append(f"\n[Pre-filled] Category: {ctx['pre_filled_category']}")

    parts.append(f"\nValid actions: {valid_actions}")
    parts.append(f"Score so far: {observation.get('score_so_far', 0):.3f}")

    if task_id == "task1":
        parts.append("\nYour action: classify this ticket.")
    elif task_id == "task2":
        completed = observation.get("completed_actions", [])
        ticket_id = ticket["id"]
        has_priority = any(ticket_id in a and "prioritize" in a for a in completed)
        if not has_priority:
            parts.append("\nYour action: prioritize this ticket (action_type=prioritize).")
        else:
            parts.append("\nYour action: route this ticket (action_type=route).")
    elif task_id == "task3":
        completed = observation.get("completed_actions", [])
        ticket_id = ticket["id"]
        done_actions = [a.split("] ")[1] for a in completed if ticket_id in a]
        if "classify" not in done_actions:
            parts.append("\nYour action: classify (action_type=classify).")
        elif "prioritize" not in done_actions:
            parts.append("\nYour action: prioritize (action_type=prioritize).")
        elif "route" not in done_actions:
            parts.append("\nYour action: route (action_type=route).")
        else:
            parts.append("\nYour action: write a full customer response (action_type=respond).")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, client: OpenAI, host: str) -> Dict[str, Any]:
    """Run one complete episode and return summary stats."""
    print(f"\n{'='*60}")
    print(f"  Running task: {task_id.upper()}")
    print(f"{'='*60}")

    obs = env_reset(task_id, host)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    total_reward = 0.0
    step = 0
    final_grader_score = None
    action_results = []

    while step < MAX_STEPS:
        if obs.get("ticket") is None:
            print("  No more tickets — episode complete.")
            break

        user_msg = build_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

        print(f"\n  Step {step + 1}: Ticket {obs.get('context', {}).get('ticket_number', '?')}"
              f" — {obs.get('ticket', {}).get('subject', '')[:50]}...")

        # Get LLM action
        raw_response = call_llm(client, messages)
        messages.append({"role": "assistant", "content": raw_response})

        action = parse_action(raw_response)
        if action is None:
            print(f"  ⚠ Failed to parse LLM response: {raw_response[:100]}")
            action = {"action_type": "skip"}

        print(f"  Action: {action.get('action_type')} "
              f"{action.get('category', action.get('priority', action.get('department', '')))}")

        # Execute action
        result = env_step(action, host)
        reward_val = result.get("reward", {}).get("value", 0.0)
        reason = result.get("reward", {}).get("reason", "")
        total_reward += reward_val
        step += 1

        print(f"  Reward: {reward_val:+.4f}  |  {reason}")

        action_results.append({
            "step": step,
            "action_type": action.get("action_type"),
            "reward": reward_val,
            "reason": reason,
        })

        # Check terminal
        if result.get("done"):
            grader_result = result.get("info", {}).get("grader_result", {})
            final_grader_score = grader_result.get("score", None)
            if final_grader_score is not None:
                print(f"\n  ✅ Episode done. Grader score: {final_grader_score:.4f}")
            break

        obs = result["observation"]
        time.sleep(0.1)  # small pause to avoid rate limiting

    # Fallback: get score from state
    if final_grader_score is None:
        state = env_state(host)
        final_grader_score = state.get("grader_scores", {}).get("final", None)

    return {
        "task_id": task_id,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "grader_score": final_grader_score,
        "action_results": action_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Customer Support Triage — OpenEnv Baseline")
    parser.add_argument("--task", default="all", help="task1 | task2 | task3 | all")
    parser.add_argument("--host", default=ENV_HOST, help="Environment host URL")
    args = parser.parse_args()

    # Validate environment variables
    if not HF_TOKEN:
        print("⚠ Warning: HF_TOKEN not set. LLM calls may fail.")
    if not API_BASE_URL:
        print("⚠ Warning: API_BASE_URL not set. Using OpenAI default.")

    # Initialize OpenAI client
    client = OpenAI(
        api_key=HF_TOKEN or "placeholder",
        base_url=API_BASE_URL,
    )

    # Health check
    try:
        resp = requests.get(f"{args.host}/health", timeout=10)
        resp.raise_for_status()
        print(f"✅ Environment healthy: {resp.json()}")
    except Exception as e:
        print(f"❌ Environment not reachable at {args.host}: {e}")
        sys.exit(1)

    # Determine tasks to run
    tasks_to_run = ["task1", "task2", "task3"] if args.task == "all" else [args.task]

    # Run episodes
    results = []
    start_time = time.time()

    for task_id in tasks_to_run:
        try:
            result = run_episode(task_id, client, args.host)
            results.append(result)
        except Exception as e:
            print(f"❌ Task {task_id} failed: {e}")
            results.append({"task_id": task_id, "error": str(e), "grader_score": 0.0})

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print("  BASELINE SCORES SUMMARY")
    print(f"{'='*60}")
    for r in results:
        score = r.get("grader_score", "N/A")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        steps = r.get("steps", "N/A")
        err = r.get("error", "")
        status = "❌ ERROR" if err else "✅"
        print(f"  {status} {r['task_id']:8s}  grader_score={score_str:6s}  steps={steps}")
        if err:
            print(f"           error: {err}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API: {API_BASE_URL}")
    print(f"{'='*60}\n")

    # Save results to JSON
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "api_base_url": API_BASE_URL,
                "elapsed_seconds": round(elapsed, 2),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
