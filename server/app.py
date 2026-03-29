from __future__ import annotations
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Enums ──────────────────────────────────────────────────────────────────
class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"
    FEATURE_REQUEST = "feature_request"

class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Department(str, Enum):
    BILLING_TEAM = "billing_team"
    TECH_SUPPORT = "tech_support"
    ACCOUNT_MANAGEMENT = "account_management"
    GENERAL_SUPPORT = "general_support"

class ActionType(str, Enum):
    CLASSIFY = "classify"
    PRIORITIZE = "prioritize"
    ROUTE = "route"
    RESPOND = "respond"
    SKIP = "skip"

# ── Pydantic Models ─────────────────────────────────────────────────────────
class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    customer_tier: str
    timestamp: str
    sla_hours: int = 24
    previous_messages: List[str] = Field(default_factory=list)

class Action(BaseModel):
    action_type: ActionType
    category: Optional[TicketCategory] = None
    priority: Optional[Priority] = None
    department: Optional[Department] = None
    response_text: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str
    partial_credit: Dict[str, float] = Field(default_factory=dict)
    is_terminal: bool = False

class Observation(BaseModel):
    ticket: Optional[Ticket] = None
    task_id: str = ""
    task_description: str = ""
    current_ticket_index: int = 0
    total_tickets: int = 0
    current_step: int = 0
    max_steps: int = 0
    completed_actions: List[str] = Field(default_factory=list)
    score_so_far: float = 0.0
    valid_action_types: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class EnvironmentState(BaseModel):
    task_id: str = ""
    episode_id: str = ""
    tickets: List[Ticket] = Field(default_factory=list)
    current_ticket_index: int = 0
    step: int = 0
    max_steps: int = 0
    total_reward: float = 0.0
    done: bool = True
    action_log: List[Dict[str, Any]] = Field(default_factory=list)
    grader_scores: Dict[str, float] = Field(default_factory=dict)

class ResetRequest(BaseModel):
    task_id: str = "task1"

# ── Ticket Data ─────────────────────────────────────────────────────────────
TASK1_TICKETS = [
    {"id":"T1-001","subject":"Charged twice this billing cycle","body":"I see two charges of $49.99. Please investigate and refund the duplicate.","customer_tier":"premium","timestamp":"2024-03-05T08:15:00Z","sla_hours":4,"correct_category":"billing","correct_priority":"high","correct_department":"billing_team"},
    {"id":"T1-002","subject":"API returning 500 errors","body":"Production hitting 500 errors on /v2/data since 6 AM. Enterprise SLA breach.","customer_tier":"enterprise","timestamp":"2024-03-05T09:02:00Z","sla_hours":1,"correct_category":"technical","correct_priority":"urgent","correct_department":"tech_support"},
    {"id":"T1-003","subject":"Cannot log in — account does not exist","body":"Login says account not found. Been a customer 2 years. Urgent access needed.","customer_tier":"basic","timestamp":"2024-03-05T10:30:00Z","sla_hours":8,"correct_category":"account","correct_priority":"high","correct_department":"account_management"},
    {"id":"T1-004","subject":"How do I export data to CSV?","body":"Looking for CSV export in Settings but cannot find it. Is it available?","customer_tier":"free","timestamp":"2024-03-05T11:45:00Z","sla_hours":48,"correct_category":"general","correct_priority":"low","correct_department":"general_support"},
    {"id":"T1-005","subject":"Would love a dark mode option","body":"Dark mode would be great for late night work. Has this been considered?","customer_tier":"basic","timestamp":"2024-03-05T13:00:00Z","sla_hours":72,"correct_category":"feature_request","correct_priority":"low","correct_department":"general_support"},
]
TASK2_TICKETS = [
    {"id":"T2-001","subject":"Complete platform outage","body":"500 users cannot access platform for 20 minutes. Enterprise SLA breach.","customer_tier":"enterprise","timestamp":"2024-03-06T07:00:00Z","sla_hours":1,"category":"technical","correct_category":"technical","correct_priority":"urgent","correct_department":"tech_support"},
    {"id":"T2-002","subject":"Need annual invoice for tax filing","body":"Need consolidated 2023 invoice for tax deadline next week.","customer_tier":"basic","timestamp":"2024-03-06T09:00:00Z","sla_hours":48,"category":"billing","correct_category":"billing","correct_priority":"medium","correct_department":"billing_team"},
    {"id":"T2-003","subject":"Want to upgrade to Enterprise plan","body":"45 users, need SSO and audit logs. Who handles Enterprise pricing?","customer_tier":"premium","timestamp":"2024-03-06T10:30:00Z","sla_hours":24,"category":"account","correct_category":"account","correct_priority":"high","correct_department":"account_management"},
]
TASK3_TICKETS = [
    {"id":"T3-001","subject":"Billing error plus API failures","body":"Overcharged $150 third month in a row. API returning 429s within quota. Premium 4 years. Will chargeback by EOD.","customer_tier":"premium","timestamp":"2024-03-07T08:00:00Z","sla_hours":2,"previous_messages":["Agent (Feb): Applied credit."],"correct_category":"billing","correct_priority":"urgent","correct_department":"billing_team","response_keywords":["apolog","billing","refund","rate limit","investigate","priorit","escalat","loyal","credit"],"response_min_words":80},
    {"id":"T3-002","subject":"Unexpected login from Russia","body":"Login from Moscow at 3AM. Changed password, enabled 2FA. Worried about data access. GDPR concern.","customer_tier":"enterprise","timestamp":"2024-03-07T09:30:00Z","sla_hours":1,"previous_messages":[],"correct_category":"account","correct_priority":"urgent","correct_department":"account_management","response_keywords":["secur","investigat","audit","escalat","2fa","gdpr","urgently","team"],"response_min_words":100},
    {"id":"T3-003","subject":"Requesting discount for renewal","body":"Renewing at $2400. 3 year customer. Competitors offer 20 percent off. Budget tight.","customer_tier":"premium","timestamp":"2024-03-07T11:00:00Z","sla_hours":48,"previous_messages":[],"correct_category":"billing","correct_priority":"high","correct_department":"account_management","response_keywords":["thank","loyal","discount","renew","team","discuss","value","appreciat"],"response_min_words":60},
]

TASKS = {
    "task1": {"id":"task1","name":"Ticket Classification","difficulty":"easy","description":"Classify each ticket into: billing, technical, account, general, or feature_request","valid_action_types":["classify","skip"],"tickets":TASK1_TICKETS,"max_steps":14,"reward_config":{"correct_classify":1.0/len(TASK1_TICKETS),"wrong_classify":-0.05}},
    "task2": {"id":"task2","name":"Priority and Routing","difficulty":"medium","description":"Set priority and route to the correct department for each ticket","valid_action_types":["prioritize","route","skip"],"tickets":TASK2_TICKETS,"max_steps":15,"reward_config":{"correct_priority":0.5/len(TASK2_TICKETS),"adjacent_priority":0.25/len(TASK2_TICKETS),"wrong_priority":-0.03,"correct_routing":0.5/len(TASK2_TICKETS),"wrong_routing":-0.03}},
    "task3": {"id":"task3","name":"Full Ticket Resolution","difficulty":"hard","description":"Classify, prioritize, route, and write a full response for each ticket","valid_action_types":["classify","prioritize","route","respond","skip"],"tickets":TASK3_TICKETS,"max_steps":20,"reward_config":{"correct_classify":0.20/len(TASK3_TICKETS),"correct_priority":0.20/len(TASK3_TICKETS),"correct_routing":0.20/len(TASK3_TICKETS),"response_quality_max":0.40/len(TASK3_TICKETS),"wrong_any":-0.02}},
}

# ── Scoring helpers ──────────────────────────────────────────────────────────
PRIORITY_ORDER = ["urgent", "high", "medium", "low"]

def priority_score(pred: str, correct: str) -> float:
    if pred == correct:
        return 1.0
    try:
        if abs(PRIORITY_ORDER.index(pred) - PRIORITY_ORDER.index(correct)) == 1:
            return 0.5
    except ValueError:
        pass
    return 0.0

def response_score(resp: str, keywords: list, min_words: int) -> float:
    if not resp or not resp.strip():
        return 0.0
    r = resp.lower()
    words = resp.split()
    length = min(1.0, len(words) / max(min_words, 1))
    kw = sum(1 for k in keywords if k in r) / max(len(keywords), 1)
    greet = any(g in r for g in ["hello", "hi ", "dear", "thank you for", "thanks for"])
    close = any(c in r for c in ["regards", "sincerely", "best", "let us know", "feel free", "happy to help"])
    prof = 0.5 * int(greet) + 0.5 * int(close)
    emp = min(1.0, sum(1 for p in ["understand", "apolog", "sorry", "frustrat", "appreciate", "concern"] if p in r) / 3)
    return round(0.25 * length + 0.40 * kw + 0.20 * prof + 0.15 * emp, 4)

# ── Environment ──────────────────────────────────────────────────────────────
class Env:
    ENV_ID = "customer-support-triage-v1"
    VERSION = "1.0.0"

    def __init__(self):
        self._task_id = None
        self._task_def = None
        self._tickets = []
        self._step = 0
        self._max_steps = 0
        self._idx = 0
        self._done = True
        self._total_reward = 0.0
        self._action_log = []
        self._grader_scores = {}
        self._ticket_counts = {}

    def reset(self, task_id: str = "task1") -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(TASKS.keys())}")
        self._task_id = task_id
        self._task_def = TASKS[task_id]
        self._tickets = deepcopy(self._task_def["tickets"])
        self._step = 0
        self._max_steps = self._task_def["max_steps"]
        self._idx = 0
        self._done = False
        self._total_reward = 0.0
        self._action_log = []
        self._grader_scores = {}
        self._ticket_counts = {i: {} for i in range(len(self._tickets))}
        return self._build_obs()

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        self._step += 1
        reward, info = self._act(action)
        self._total_reward = round(self._total_reward + reward.value, 6)
        done = self._idx >= len(self._tickets) or self._step >= self._max_steps
        if done:
            self._done = True
            gs = self._grade()
            self._grader_scores["final"] = gs["score"]
            info["grader_result"] = gs
            reward = Reward(value=0.0, reason="Episode complete", is_terminal=True, partial_credit={"final_score": gs["score"]})
        return StepResult(observation=self._build_obs(), reward=reward, done=done, info=info)

    def state(self) -> EnvironmentState:
        tickets = [Ticket(id=t["id"], subject=t["subject"], body=t["body"], customer_tier=t["customer_tier"], timestamp=t["timestamp"], sla_hours=t.get("sla_hours", 24), previous_messages=t.get("previous_messages", [])) for t in self._tickets]
        return EnvironmentState(task_id=self._task_id or "", episode_id="", tickets=tickets, current_ticket_index=self._idx, step=self._step, max_steps=self._max_steps, total_reward=self._total_reward, done=self._done, action_log=self._action_log, grader_scores=self._grader_scores)

    def _build_obs(self) -> Observation:
        task_def = self._task_def or {}
        ticket = None
        ctx: Dict[str, Any] = {}
        if self._idx < len(self._tickets):
            t = self._tickets[self._idx]
            ticket = Ticket(id=t["id"], subject=t["subject"], body=t["body"], customer_tier=t["customer_tier"], timestamp=t["timestamp"], sla_hours=t.get("sla_hours", 24), previous_messages=t.get("previous_messages", []))
            ctx = {"ticket_number": self._idx + 1, "total_tickets": len(self._tickets)}
            if self._task_id == "task2":
                ctx["pre_filled_category"] = t.get("category", "")
        return Observation(ticket=ticket, task_id=self._task_id or "", task_description=task_def.get("description", ""), current_ticket_index=self._idx, total_tickets=len(self._tickets), current_step=self._step, max_steps=self._max_steps, completed_actions=[f'[{a["ticket_id"]}] {a["action_type"]}' for a in self._action_log], score_so_far=self._total_reward, valid_action_types=task_def.get("valid_action_types", []), context=ctx)

    def _act(self, action: Action):
        if self._idx >= len(self._tickets):
            return Reward(value=0.0, reason="No more tickets"), {}
        raw = self._tickets[self._idx]
        cfg = self._task_def["reward_config"]
        at = action.action_type
        ad = action.model_dump()
        ad["ticket_id"] = raw["id"]

        if at == ActionType.SKIP:
            self._action_log.append(ad)
            self._idx += 1
            return Reward(value=-0.02, reason="Skipped ticket"), {"skipped": raw["id"]}

        self._action_log.append(ad)
        counts = self._ticket_counts.setdefault(self._idx, {})
        counts[at.value] = counts.get(at.value, 0) + 1

        if at == ActionType.CLASSIFY:
            pred = action.category.value if action.category else ""
            exp = raw["correct_category"]
            ok = pred == exp
            rv = cfg["correct_classify"] if ok else cfg.get("wrong_classify", -0.05)
            r = Reward(value=rv, reason=f"{'correct' if ok else 'wrong'} category '{pred}'", partial_credit={"correct": float(ok)})
            if self._task_id == "task1":
                self._idx += 1
            return r, {"predicted": pred, "expected": exp}

        if at == ActionType.PRIORITIZE:
            pred = action.priority.value if action.priority else ""
            exp = raw["correct_priority"]
            ps = priority_score(pred, exp)
            rv = cfg.get("correct_priority", 0.1) * ps if ps > 0 else cfg.get("wrong_priority", -0.03)
            return Reward(value=rv, reason=f"priority score {ps:.1f}", partial_credit={"score": ps}), {}

        if at == ActionType.ROUTE:
            pred = action.department.value if action.department else ""
            exp = raw["correct_department"]
            ok = pred == exp
            rv = cfg.get("correct_routing", 0.1) if ok else cfg.get("wrong_routing", -0.03)
            r = Reward(value=rv, reason=f"{'correct' if ok else 'wrong'} routing", partial_credit={"correct": float(ok)})
            if self._task_id == "task2" and counts.get("prioritize", 0) > 0 and counts.get("route", 0) > 0:
                self._idx += 1
            return r, {}

        if at == ActionType.RESPOND:
            resp = action.response_text or ""
            rs = response_score(resp, raw.get("response_keywords", []), raw.get("response_min_words", 60))
            rv = round(cfg.get("response_quality_max", 0.133) * rs, 6)
            r = Reward(value=rv, reason=f"response quality {rs:.2f}", partial_credit={"quality": rs})
            if self._task_id == "task3" and all(counts.get(a, 0) > 0 for a in ["classify", "prioritize", "route", "respond"]):
                self._idx += 1
            return r, {}

        return Reward(value=0.0, reason="unknown action"), {}

    def _grade(self) -> Dict[str, Any]:
        cl = [a for a in self._action_log if a.get("action_type") == "classify"]
        pr = [a for a in self._action_log if a.get("action_type") == "prioritize"]
        ro = [a for a in self._action_log if a.get("action_type") == "route"]
        re = [a for a in self._action_log if a.get("action_type") == "respond"]
        tickets = self._tickets
        n = max(len(tickets), 1)

        if self._task_id == "task1":
            correct = sum(1 for i, t in enumerate(tickets) if i < len(cl) and cl[i].get("category") == t["correct_category"])
            return {"score": round(correct / n, 4), "reason": f"{correct}/{len(tickets)} correct"}

        if self._task_id == "task2":
            ps = [priority_score(pr[i].get("priority", "") if i < len(pr) else "", t["correct_priority"]) for i, t in enumerate(tickets)]
            rs = [1.0 if i < len(ro) and ro[i].get("department") == t["correct_department"] else 0.0 for i, t in enumerate(tickets)]
            return {"score": round(0.5 * sum(ps) / n + 0.5 * sum(rs) / n, 4), "reason": "priority+routing"}

        if self._task_id == "task3":
            scores = []
            for i, t in enumerate(tickets):
                cs = 1.0 if i < len(cl) and cl[i].get("category") == t["correct_category"] else 0.0
                ps = priority_score(pr[i].get("priority", "") if i < len(pr) else "", t["correct_priority"])
                ds = 1.0 if i < len(ro) and ro[i].get("department") == t["correct_department"] else 0.0
                resp = re[i].get("response_text", "") if i < len(re) else ""
                rs = response_score(resp, t.get("response_keywords", []), t.get("response_min_words", 60))
                scores.append(0.20 * cs + 0.20 * ps + 0.20 * ds + 0.40 * rs)
            return {"score": round(sum(scores) / n, 4), "reason": "full resolution"}

        return {"score": 0.0, "reason": "unknown task"}


_env = Env()

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Customer Support Triage OpenEnv", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "env_id": Env.ENV_ID, "version": Env.VERSION}

@app.get("/tasks")
def get_tasks():
    return {"tasks": [{"id": t["id"], "name": t["name"], "difficulty": t["difficulty"], "num_tickets": len(t["tickets"])} for t in TASKS.values()]}

@app.post("/reset")
def reset(body: ResetRequest = None):
    task_id = (body.task_id if body else None) or "task1"
    try:
        obs = _env.reset(task_id=task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: Action):
    try:
        result = _env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/state")
def state():
    return _env.state().model_dump()

@app.get("/")
def root():
    return {"message": "Customer Support Triage OpenEnv", "docs": "/docs", "health": "/health", "tasks": "/tasks"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
