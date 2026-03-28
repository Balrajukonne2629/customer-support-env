"""
FastAPI application exposing the CustomerSupportEnv via HTTP.
Deployed to Hugging Face Spaces — tagged openenv.

Endpoints:
  POST /reset          → Observation
  POST /step           → StepResult
  GET  /state          → EnvironmentState
  GET  /tasks          → list of available tasks
  GET  /health         → liveness check
  GET  /               → HTML documentation page
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment import (
    Action,
    CustomerSupportEnv,
    EnvironmentState,
    Observation,
    StepResult,
    list_tasks,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Support Triage — OpenEnv",
    description=(
        "An OpenEnv environment for training and evaluating AI agents "
        "on real-world customer support ticket management."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# One global environment instance (stateful per session)
# For production multi-user, use a session map keyed by session_id.
_env = CustomerSupportEnv()


# ---------------------------------------------------------------------------
# Request/response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    tasks = list_tasks()
    task_rows = "".join(
        f"<tr><td>{t['id']}</td><td>{t['name']}</td>"
        f"<td>{t['difficulty']}</td><td>{t['num_tickets']}</td></tr>"
        for t in tasks
    )
    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Customer Support Triage — OpenEnv</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1 {{ color: #1a56db; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }}
    th {{ background: #f9fafb; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
    .easy {{ background: #d1fae5; color: #065f46; }}
    .medium {{ background: #fef3c7; color: #92400e; }}
    .hard {{ background: #fee2e2; color: #991b1b; }}
  </style>
</head>
<body>
  <h1>🎫 Customer Support Triage — OpenEnv</h1>
  <p>
    A real-world OpenEnv environment where AI agents manage a customer support ticket queue.
    Agents must classify, prioritize, route, and respond to tickets — skills directly
    applicable to enterprise SaaS support automation.
  </p>

  <h2>Tasks</h2>
  <table>
    <tr><th>ID</th><th>Name</th><th>Difficulty</th><th>Tickets</th></tr>
    {task_rows}
  </table>

  <h2>Quick Start</h2>
  <pre><code>
# Start an episode
POST /reset  {{"task_id": "task1"}}

# Take an action
POST /step   {{"action_type": "classify", "category": "billing"}}

# Check state
GET  /state
  </code></pre>

  <p>
    📖 <a href="/docs">Interactive API docs (Swagger)</a> &nbsp;|&nbsp;
    📋 <a href="/redoc">ReDoc</a>
  </p>
</body>
</html>"""


@app.get("/health")
def health():
    return {"status": "ok", "env_id": CustomerSupportEnv.ENV_ID, "version": CustomerSupportEnv.VERSION}


@app.get("/tasks")
def get_tasks():
    return {"tasks": list_tasks()}


@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest):
    try:
        obs = _env.reset(task_id=body.task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state", response_model=EnvironmentState)
def state():
    return _env.state()


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
