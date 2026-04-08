"""
inference.py — Customer Support Triage OpenEnv
Baseline inference script using OpenAI client.

Required env vars:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — API key

Emits structured stdout:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
ENV_HOST: str = os.getenv("ENV_HOST", "http://localhost:7860")

TEMPERATURE: float = 0.0
MAX_TOKENS: int = 512
MAX_STEPS: int = 25
SUCCESS_THRESHOLD: float = 0.5
BENCHMARK: str = "customer-support-triage-v1"
TASKS: List[str] = ["task1", "task2", "task3"]

# ── Structured Logging ───────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_safe = str(action).replace("\n", " ")[:80]
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Environment Client ───────────────────────────────────────────────────────
def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_HOST}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_HOST}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ── LLM Agent ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support agent AI.

For each ticket, respond with ONLY a valid JSON object — no markdown, no explanation:

For task1 (classify):
{"action_type": "classify", "category": "<billing|technical|account|general|feature_request>"}

For task2 (prioritize first, then route):
{"action_type": "prioritize", "priority": "<urgent|high|medium|low>"}
{"action_type": "route", "department": "<billing_team|tech_support|account_management|general_support>"}

For task3 (classify, prioritize, route, then respond):
{"action_type": "classify", "category": "<billing|technical|account|general|feature_request>"}
{"action_type": "prioritize", "priority": "<urgent|high|medium|low>"}
{"action_type": "route", "department": "<billing_team|tech_support|account_management|general_support>"}
{"action_type": "respond", "response_text": "<professional reply, min 80 words>"}

Priority guide: urgent=outage/security, high=major impact, medium=moderate, low=minor
Department guide: billing_team=payments, tech_support=bugs/API, account_management=access/upgrades, general_support=questions
"""

def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""

def parse_action(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None

def build_prompt(obs: Dict[str, Any]) -> str:
    ticket = obs.get("ticket")
    if not ticket:
        return "No ticket available."
    task_id = obs.get("task_id", "")
    ctx = obs.get("context", {})
    completed = obs.get("completed_actions", [])
    ticket_id = ticket.get("id", "")

    parts = [
        f"TASK: {task_id.upper()}",
        f"Ticket {ctx.get('ticket_number','?')}/{ctx.get('total_tickets','?')}",
        f"Subject: {ticket.get('subject','')}",
        f"Customer tier: {ticket.get('customer_tier','')}",
        f"SLA hours: {ticket.get('sla_hours','N/A')}",
        f"Message: {ticket.get('body','')}",
    ]

    prev = ticket.get("previous_messages", [])
    if prev:
        parts.append("History: " + " | ".join(prev))

    if ctx.get("pre_filled_category"):
        parts.append(f"Category already set: {ctx['pre_filled_category']}")

    done_actions = [a.split("] ")[1] for a in completed if ticket_id in a]

    if task_id == "task1":
        parts.append("ACTION NEEDED: classify this ticket. Output JSON with action_type=classify")
    elif task_id == "task2":
        if "prioritize" not in done_actions:
            parts.append("ACTION NEEDED: prioritize this ticket. Output JSON with action_type=prioritize")
        else:
            parts.append("ACTION NEEDED: route this ticket. Output JSON with action_type=route")
    elif task_id == "task3":
        if "classify" not in done_actions:
            parts.append("ACTION NEEDED: classify. Output JSON with action_type=classify")
        elif "prioritize" not in done_actions:
            parts.append("ACTION NEEDED: prioritize. Output JSON with action_type=prioritize")
        elif "route" not in done_actions:
            parts.append("ACTION NEEDED: route. Output JSON with action_type=route")
        else:
            parts.append("ACTION NEEDED: respond with a professional, empathetic reply (min 80 words). Output JSON with action_type=respond and response_text")

    return "\n".join(parts)

def fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a safe default action if LLM fails."""
    task_id = obs.get("task_id", "task1")
    completed = obs.get("completed_actions", [])
    ticket = obs.get("ticket", {})
    ticket_id = ticket.get("id", "") if ticket else ""
    done_actions = [a.split("] ")[1] for a in completed if ticket_id in a]

    if task_id == "task1":
        return {"action_type": "classify", "category": "general"}
    elif task_id == "task2":
        if "prioritize" not in done_actions:
            return {"action_type": "prioritize", "priority": "medium"}
        return {"action_type": "route", "department": "general_support"}
    elif task_id == "task3":
        if "classify" not in done_actions:
            return {"action_type": "classify", "category": "general"}
        elif "prioritize" not in done_actions:
            return {"action_type": "prioritize", "priority": "medium"}
        elif "route" not in done_actions:
            return {"action_type": "route", "department": "general_support"}
        else:
            return {"action_type": "respond", "response_text": "Thank you for contacting us. We understand your concern and will investigate this matter promptly. Our team will review your case and get back to you as soon as possible. We apologize for any inconvenience caused and appreciate your patience. Please let us know if you need any further assistance."}
    return {"action_type": "skip"}

# ── Episode Runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str, client: OpenAI) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if not obs.get("ticket"):
                break

            user_msg = build_prompt(obs)
            messages.append({"role": "user", "content": user_msg})

            raw = call_llm(client, messages)
            messages.append({"role": "assistant", "content": raw or "null"})

            action = parse_action(raw)
            if action is None:
                action = fallback_action(obs)

            action_str = f"{action.get('action_type','?')}:{action.get('category', action.get('priority', action.get('department', action.get('response_text','')[:20] if action.get('response_text') else '')))}"

            try:
                result = env_step(action)
                reward = float(result.get("reward", {}).get("value", 0.0))
                done = bool(result.get("done", False))
                error = None
                obs = result.get("observation", obs)
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)[:80]

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                # Extract grader score
                info = result.get("info", {}) if "result" in dir() else {}
                grader = info.get("grader_result", {})
                score = float(grader.get("score", sum(rewards)))
                score = min(max(score, 0.0), 1.0)
                break

            time.sleep(0.05)

        if not rewards:
            score = 0.0
        elif score == 0.0:
            score = min(max(sum(rewards), 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score = 0.0
        success = False
        if steps_taken == 0:
            steps_taken = 1
            rewards = [0.0]
            log_step(step=1, action="error", reward=0.0, done=True, error=str(e)[:80])

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards if rewards else [0.0])

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Validate env vars
    if not HF_TOKEN:
        print("[DEBUG] Warning: HF_TOKEN not set", flush=True)

    # Init OpenAI client
    client = OpenAI(
        api_key=HF_TOKEN or "placeholder",
        base_url=API_BASE_URL,
    )

    # Health check with retries
    for attempt in range(5):
        try:
            resp = requests.get(f"{ENV_HOST}/health", timeout=10)
            if resp.status_code == 200:
                print(f"[DEBUG] Environment healthy: {resp.json()}", flush=True)
                break
        except Exception as e:
            print(f"[DEBUG] Health check attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(3)
    else:
        print("[DEBUG] Environment not reachable — running anyway", flush=True)

    # Run all 3 tasks
    for task_id in TASKS:
        try:
            run_episode(task_id, client)
        except Exception as e:
            print(f"[DEBUG] Task {task_id} crashed: {e}", flush=True)
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="error", reward=0.0, done=True, error=str(e)[:80])
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])

        time.sleep(1)


if __name__ == "__main__":
    main()
