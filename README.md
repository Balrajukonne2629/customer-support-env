# 🎫 Customer Support Triage — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](https://openenv.ai)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![HF Spaces](https://img.shields.io/badge/HF%20Spaces-deployed-yellow)](https://huggingface.co/spaces)

An **OpenEnv-compliant environment** for training and evaluating AI agents on **real-world customer support ticket management** — one of the most common and high-value automation tasks in enterprise SaaS.

---

## Motivation

Every SaaS company processes thousands of support tickets daily. Today this requires expensive human agents who read, classify, prioritize, route, and respond to each ticket. An AI agent that can do this well has **immediate economic value** — and there is currently no standardized benchmark environment to develop and evaluate such agents.

This environment fills that gap.

---

## Environment Description

An AI agent manages an incoming queue of realistic customer support tickets for a fictional SaaS company. Tickets vary in category (billing, technical, account, general, feature requests), urgency (from routine questions to production outages), customer tier (free to enterprise), and emotional tone (calm to highly frustrated).

The agent must demonstrate a full support workflow:
1. **Understand** the ticket content
2. **Classify** the issue type
3. **Assess priority** based on business impact
4. **Route** to the correct team
5. **Compose** a professional, empathetic response

---

## Tasks

| ID | Name | Difficulty | Tickets | Description |
|----|------|-----------|---------|-------------|
| `task1` | Ticket Classification | 🟢 Easy | 7 | Classify each ticket into the correct category |
| `task2` | Priority & Routing | 🟡 Medium | 5 | Assign priority level and route to correct department |
| `task3` | Full Ticket Resolution | 🔴 Hard | 3 | Classify + prioritize + route + write full response |

### Task 1 — Ticket Classification (Easy)

The agent reads each ticket and assigns it to one of five categories:
- `billing` — payment, invoice, refund, charge issues
- `technical` — API errors, bugs, performance problems, outages
- `account` — login failures, access issues, data loss, account settings
- `general` — how-to questions, documentation requests
- `feature_request` — requests for new product features

**Scoring:** Fraction of tickets correctly classified (exact match, 0.0–1.0)

### Task 2 — Priority & Routing (Medium)

Categories are pre-filled. The agent must:
1. **Prioritize:** `urgent` | `high` | `medium` | `low`
   - Accounts for customer tier, SLA hours, business impact
   - Adjacent priority levels earn partial credit (0.5)
2. **Route:** `billing_team` | `tech_support` | `account_management` | `general_support`

**Scoring:** 0.5 × priority_accuracy + 0.5 × routing_accuracy (0.0–1.0)

### Task 3 — Full Ticket Resolution (Hard)

Complex, multi-issue tickets requiring all four actions per ticket:
- **Classify** (20%) — identify the primary issue category
- **Prioritize** (20%) — assess business-impact-based urgency
- **Route** (20%) — direct to appropriate team
- **Respond** (40%) — write a complete, professional customer reply

Response quality is scored on four dimensions:
| Dimension | Weight | Criteria |
|-----------|--------|----------|
| Length adequacy | 25% | Meets minimum word count |
| Keyword coverage | 40% | Addresses key issue topics |
| Professionalism | 20% | Has greeting + closing |
| Empathy | 15% | Uses empathetic language markers |

**Scoring:** Average full-resolution score across tickets (0.0–1.0)

---

## Observation Space

```json
{
  "ticket": {
    "id": "T3-001",
    "subject": "Multiple issues: billing error + API failures",
    "body": "I am extremely frustrated...",
    "customer_tier": "premium",
    "timestamp": "2024-03-07T08:00:00Z",
    "sla_hours": 2,
    "previous_messages": ["Agent (Feb): Apologies..."]
  },
  "task_id": "task3",
  "task_description": "...",
  "current_ticket_index": 0,
  "total_tickets": 3,
  "current_step": 0,
  "max_steps": 20,
  "completed_actions": [],
  "score_so_far": 0.0,
  "valid_action_types": ["classify", "prioritize", "route", "respond", "skip"],
  "context": {"ticket_number": 1, "total_tickets": 3}
}
```

## Action Space

```json
// Classify a ticket
{"action_type": "classify", "category": "billing"}

// Set priority
{"action_type": "prioritize", "priority": "urgent"}

// Route to department
{"action_type": "route", "department": "billing_team"}

// Write response
{"action_type": "respond", "response_text": "Dear Customer, I sincerely apologize..."}

// Skip (with penalty)
{"action_type": "skip"}
```

## Reward Function

The reward function provides **continuous shaped signals** throughout the episode:

| Event | Reward |
|-------|--------|
| Correct classification | `+1/N` (N = tickets in task) |
| Wrong classification | `-0.05` |
| Correct priority (exact) | `+0.5/N` |
| Adjacent priority | `+0.25/N` |
| Wrong priority | `-0.03` |
| Correct routing | `+0.5/N` |
| Wrong routing | `-0.03` |
| Response quality | `0.0–0.133` per ticket (proportional to quality) |
| Skip | `-0.02` |

**No sparse-only rewards** — every action provides signal, enabling stable RL training.

---

## Baseline Scores

Evaluated with `gpt-4o-mini` (temperature=0.0):

| Task | Score | Notes |
|------|-------|-------|
| task1 (Classification) | **0.71** | Good on billing/technical, confuses account/general |
| task2 (Priority & Routing) | **0.68** | Priority harder than routing; partial credit helps |
| task3 (Full Resolution) | **0.52** | Response quality is the bottleneck |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode: `{"task_id": "task1"}` |
| `POST` | `/step` | Take action: `Action` object |
| `GET` | `/state` | Full environment state |
| `GET` | `/tasks` | List available tasks |
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Swagger UI |

---

## Setup & Usage

### Option 1: Docker (Recommended)

```bash
git clone https://huggingface.co/spaces/<your-username>/customer-support-triage
cd customer-support-triage

# Build and run
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env

# Verify
curl http://localhost:7860/health
```

### Option 2: Local Python

```bash
pip install -r requirements.txt
python app.py
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_HOST="http://localhost:7860"

python inference.py              # all tasks
python inference.py --task task1  # single task
```

### Quick API Test

```bash
# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# Classify ticket
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "category": "billing"}'

# Check state
curl http://localhost:7860/state
```

---

## Project Structure

```
customer-support-env/
├── environment/
│   ├── __init__.py       # Package exports
│   ├── models.py         # Pydantic typed models (Observation, Action, Reward, ...)
│   ├── data.py           # Curated realistic ticket dataset (15 tickets across 3 tasks)
│   ├── graders.py        # Deterministic per-task graders (0.0–1.0)
│   ├── tasks.py          # Task definitions with metadata and reward configs
│   └── env.py            # Core CustomerSupportEnv class
├── app.py                # FastAPI HTTP server for HF Spaces
├── inference.py          # Baseline inference script (OpenAI client)
├── openenv.yaml          # OpenEnv metadata and spec
├── Dockerfile            # Containerization config
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Design Decisions

**Why customer support?**
It is one of the most common and costly LLM deployment use cases, yet no standardized RL environment exists. Immediately useful for any company building support automation.

**Why shaped rewards over sparse?**
Sparse end-of-episode rewards make RL training unstable. Every correct action gives signal, enabling faster convergence and interpretability during training.

**Why heuristic response grading?**
Determinism and reproducibility are requirements. An LLM-based grader would produce different scores across runs. The heuristic (length + keyword coverage + professionalism + empathy) correlates well with human quality ratings while being fully reproducible.

**Why partial credit for priority?**
Missing priority by one level (urgent→high) is much less harmful than being off by three (urgent→low). Partial credit better reflects real-world impact and gives learning algorithms smoother gradient signal.

---

## License

MIT
